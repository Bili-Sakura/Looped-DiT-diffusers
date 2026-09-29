"""DPG-Bench scoring with the mPLUG-large VQA model (the official protocol).

Every prompt has a list of yes/no questions with dependencies; each of the 4
samples in a prompt's 2x2 grid is scored separately (a question whose parent was
answered "no" counts as failed) and the prompt score is the mean over samples.

    python -m looped_dit.eval.dpg --image-dir OUT/dpg_images --data Jialuo21/DPG-Bench --out-dir OUT/dpg_results

Needs `modelscope` (and its mPLUG dependencies); scoring is split over all visible GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .benchmarks import load_dpg_rows


def load_questions(data: str) -> dict[str, dict]:
    questions: dict[str, dict] = {}
    for row in load_dpg_rows(data):
        item = questions.setdefault(row["item_id"], {"question": {}, "dependency": {}, "category": {}})
        qid = int(row["proposition_id"])
        item["question"][qid] = row["question_natural_language"]
        item["dependency"][qid] = [int(d) for d in str(row["dependency"]).split(",")]
        item["category"][qid] = f"{row['category_broad']} - {row['category_detailed']}"
    return questions


def score_grids(paths: list[str], questions: dict[str, dict], mplug: str, resolution: int, gpu: str | None):
    """Returns [(image name, per-sample scores, {category: [0/1 answers of the last sample]})]."""
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    vqa = pipeline(Tasks.visual_question_answering, model=mplug, device="gpu" if torch.cuda.is_available() else "cpu")
    r = resolution
    crops = [(0, 0, r, r), (r, 0, 2 * r, r), (0, r, r, 2 * r), (r, r, 2 * r, 2 * r)]
    results = []
    for n, path in enumerate(paths, 1):
        item = questions[Path(path).stem]
        grid = Image.open(path).convert("RGB")
        scores, answers = [], {}
        for crop in crops:
            image = grid.crop(crop)
            answers = {qid: float(vqa({"image": image, "question": q})["text"] == "yes") for qid, q in item["question"].items()}
            passed = dict(answers)
            for qid, parents in item["dependency"].items():  # in question order, so failures cascade
                if any(p != 0 and passed[p] == 0 for p in parents):
                    passed[qid] = 0.0
            scores.append(sum(passed.values()) / len(passed))
        categories = defaultdict(list)
        for qid, value in answers.items():  # raw answers of the last sample, as in the official script
            categories[item["category"][qid]].append(value)
        results.append((Path(path).name, scores, dict(categories)))
        if n % 50 == 0:
            print(f"[dpg] {n}/{len(paths)}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Score DPG-Bench grids with mPLUG.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--data", default="Jialuo21/DPG-Bench")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mplug", default="iic/mplug_visual-question-answering_coco_large_en")
    parser.add_argument("--resolution", type=int, default=512, help="size of one sample in the grid")
    args = parser.parse_args()

    questions = load_questions(args.data)
    paths = [str(Path(args.image_dir) / f"{item_id}.png") for item_id in questions]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(paths)} grids are missing, e.g. {missing[0]}")

    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(torch.cuda.device_count())))).split(",")
    gpus = [g for g in gpus if g] or [None]
    with get_context("spawn").Pool(len(gpus)) as pool:
        parts = pool.starmap(
            score_grids, [(paths[i :: len(gpus)], questions, args.mplug, args.resolution, gpu) for i, gpu in enumerate(gpus)]
        )
    results = [row for part in parts for row in part]

    per_prompt = {name: float(np.mean(scores)) for name, scores, _ in results}
    categories: dict[str, list[float]] = defaultdict(list)
    for _, _, cats in results:
        for category, values in cats.items():
            categories[category].extend(values)
    l1: dict[str, list[float]] = defaultdict(list)
    for category, values in categories.items():
        l1[category.split("-")[0].strip()].extend(values)
    summary = {
        "benchmark": "dpg",
        "overall": 100 * float(np.mean(list(per_prompt.values()))),
        "num_prompts": len(per_prompt),
        "l1_categories": {k: 100 * float(np.mean(v)) for k, v in sorted(l1.items())},
        "l2_categories": {k: 100 * float(np.mean(v)) for k, v in sorted(categories.items())},
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_prompt.json").write_text(json.dumps(per_prompt, indent=1))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
