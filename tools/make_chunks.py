"""Write pretraining data as tensor chunks (chunk_000000.pt, ...).

A chunk holds --chunk-size samples: uint8 images resized (shorter side) and
center-cropped to --image-size, FLAN-T5 token ids and attention masks padded to
--prompt-length, and the captions. An interrupted run continues where it stopped.

CC12M with LLaVA-NeXT recaptions, downloaded as WebDataset shards (see README):
    python tools/make_chunks.py --webdataset data/cc12m_wds --out data/cc12m_chunks

FLUX-Reason-6M images with the i1 recaptions (parquet files with five captions per image id):
    python tools/make_chunks.py --hf-dataset LucasFang/FLUX-Reason-6M --id-column id \
        --captions data/i1-captions/fluxreason --out data/fluxreason_chunks
"""

from __future__ import annotations

import argparse
import io
import json
import zlib
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torchvision import transforms
from transformers import AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True


def from_webdataset(directory: str):
    """(key, encoded image, caption) from WebDataset shards."""
    import webdataset as wds

    shards = sorted(str(p) for p in Path(directory).rglob("*.tar"))
    for sample in wds.WebDataset(shards, handler=wds.warn_and_continue, empty_check=False, shardshuffle=False):
        image = next((sample[ext] for ext in ("jpg", "jpeg", "png", "webp") if ext in sample), None)
        yield sample["__key__"], image, sample.get("txt", b"").decode("utf-8", errors="replace")


def from_hf_dataset(name: str, split: str, image_column: str, caption_column: str, id_column: str | None):
    """(key, encoded image, caption) from a streamed Hugging Face dataset."""
    from datasets import Image as ImageFeature, load_dataset

    dataset = load_dataset(name, split=split, streaming=True).cast_column(image_column, ImageFeature(decode=False))
    for row in dataset:
        yield (str(row[id_column]) if id_column else None), row[image_column]["bytes"], row.get(caption_column) or ""


def load_captions(path: str) -> dict[str, str]:
    """Sample id -> caption, from JSON files ({id: caption}) or from parquet files with a `key`
    column and several caption columns (the i1 recaptions), of which each image gets one."""
    root = Path(path)
    files = sorted(p for p in root.rglob("*") if p.suffix in (".json", ".parquet")) if root.is_dir() else [root]
    captions: dict[str, str] = {}
    for file in files:
        if file.suffix == ".json":
            captions.update(json.loads(file.read_text(encoding="utf-8")))
            continue
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(file)
        columns = [name for name in parquet.schema_arrow.names if name.startswith("caption")]
        for batch in parquet.iter_batches(columns=["key", *columns]):
            rows = batch.to_pydict()
            for i, key in enumerate(rows["key"]):
                options = [rows[name][i] for name in columns if rows[name][i]]
                if options:  # a random caption, but the same one on every run
                    captions[str(key)] = options[zlib.crc32(str(key).encode()) % len(options)]
    return captions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--webdataset", help="directory of WebDataset shards with an image and a .txt caption")
    source.add_argument("--hf-dataset", help="Hugging Face dataset with an image column (streamed)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--caption-column", default="caption")
    parser.add_argument("--id-column", help="column matching the keys of --captions")
    parser.add_argument("--captions", help="captions by sample id: JSON files ({id: caption}) or i1 parquet files, "
                                           "a file or a directory of them")
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--tokenizer", default="google/flan-t5-large")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=args.prompt_length)
    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_size),
            transforms.PILToTensor(),
        ]
    )
    caption_map = load_captions(args.captions) if args.captions else None
    samples = (
        from_webdataset(args.webdataset)
        if args.webdataset
        else from_hf_dataset(args.hf_dataset, args.split, args.image_column, args.caption_column, args.id_column)
    )
    progress_file = out / "progress.json"
    done = json.loads(progress_file.read_text()) if progress_file.exists() else {"chunks": 0, "consumed": 0}
    chunk, consumed, images, captions = done["chunks"], 0, [], []
    for key, image, caption in samples:
        consumed += 1
        if consumed <= done["consumed"]:  # already in an earlier chunk
            continue
        if caption_map is not None:
            if key not in caption_map:
                continue
            caption = caption_map[key]
        try:
            images.append(transform(Image.open(io.BytesIO(image)).convert("RGB")))
        except Exception as error:
            print(f"skipping {key}: {error}", flush=True)
            continue
        captions.append(str(caption))
        if len(images) == args.chunk_size:
            tokens = tokenizer(captions, max_length=args.prompt_length, padding="max_length", truncation=True, return_tensors="pt")
            path = out / f"chunk_{chunk:06d}.pt"
            payload = {"pixel_values": torch.stack(images), "input_ids": tokens.input_ids,
                       "attention_mask": tokens.attention_mask, "caption": captions}
            torch.save(payload, path.with_suffix(".tmp"))
            path.with_suffix(".tmp").rename(path)
            chunk, images, captions = chunk + 1, [], []
            progress_file.write_text(json.dumps({"chunks": chunk, "consumed": consumed}))
            print(f"wrote {path}", flush=True)
    print(f"done: {chunk} chunks; {len(images)} trailing samples did not fill a chunk and were dropped")


if __name__ == "__main__":
    main()
