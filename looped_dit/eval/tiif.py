"""TIIF-Bench scoring with GPT-4o, following the official protocol on the short prompts.

    OPENAI_API_KEY=... python -m looped_dit.eval.tiif --data eval_assets/tiif/data \
        --eval-script eval_assets/tiif/eval_with_vlm.py --image-dir OUT/tiif_images --out-dir OUT/tiif_results

The judge answers all yes/no questions of an image in one request, phrased with one of the
three official templates (read verbatim from eval/eval_with_vlm.py of the TIIF-Bench
repository), at temperature 1. A prompt type scores the fraction of correct answers over all
its questions, and the overall score is the mean over the 39 prompt types. The judge can be
any OpenAI-compatible endpoint (--api-base); the key is read from $OPENAI_API_KEY.
Judgments are cached in results.jsonl, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import requests

from .benchmarks import Prompt, image_path, load_prompts
from .judge import write_jsonl

SYSTEM = "You are a professional image critic."
# sha256 of eval/eval_with_vlm.py in github.com/A113N-W3I/TIIF-Bench used for the paper.
TIIF_SCRIPT_SHA256 = "4da4e136db150f929171d3c47e3ca46c010cd2d1e1b212498fd64098d04ac76a"


def load_templates(script: str | Path) -> list[str]:
    source = Path(script).read_text(encoding="utf-8")
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != TIIF_SCRIPT_SHA256:
        print(f"[tiif] warning: {script} differs from the eval_with_vlm.py used for the paper", flush=True)
    found = dict(re.findall(r"^(raw_prompt(?:_\d)?) = '''(.*?)'''", source, re.DOTALL | re.MULTILINE))
    names = ("raw_prompt", "raw_prompt_1", "raw_prompt_2")
    if any(name not in found for name in names):
        raise ValueError(f"{script} does not define the three TIIF question templates")
    return [found[name] for name in names]


def parse_answers(response: str, count: int) -> list[str]:
    answers = [m.group(1).lower() for line in response.splitlines() if (m := re.match(r"^(yes|no)\b", line.strip(), re.IGNORECASE))]
    if len(answers) != count:
        raise ValueError(f"{len(answers)} answers for {count} questions")
    return answers


def ask(args: argparse.Namespace, api_key: str, text: str, image: Path) -> str:
    image_url = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode("ascii")
    payload = {
        "model": args.judge_model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": image_url}}]},
        ],
        "temperature": 1.0,
    }
    endpoint = args.api_base.rstrip("/") + "/chat/completions"
    response = requests.post(endpoint, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=args.timeout)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def judge(args: argparse.Namespace, api_key: str, templates: list[str], prompt: Prompt, sample: int, image: Path) -> dict:
    """Judge one image; a failed or unparseable answer is retried with a freshly drawn template."""
    key = f"{prompt.group}/{prompt.item_id}/{sample}"
    questions = [q.strip() for q in prompt.metadata["yn_question_list"]]
    reference = [str(a).strip().lower() for a in prompt.metadata["yn_answer_list"]]
    rng = random.Random(args.seed + int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
    error = None
    for _ in range(args.retries):
        try:
            response = ask(args, api_key, rng.choice(templates).replace("##YNQuestions##", "\n".join(questions)), image)
            answers = parse_answers(response, len(questions))
            return {"key": key, "group": prompt.group, "item_id": prompt.item_id, "sample": sample,
                    "answers": answers, "reference": reference, "response": response}
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            error = exc
    raise RuntimeError(f"{key}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score TIIF-Bench images with GPT-4o.")
    parser.add_argument("--data", required=True, help="data/ directory of the TIIF-Bench repository")
    parser.add_argument("--eval-script", required=True, help="eval/eval_with_vlm.py of the TIIF-Bench repository")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1234, help="seeds the template choice")
    parser.add_argument("--max-failed", type=int, default=30, help="images without a usable judgment tolerated in the summary")
    args = parser.parse_args()
    templates = load_templates(args.eval_script)

    tasks = []
    for prompt in load_prompts("tiif", args.data):
        for sample in range(args.samples_per_prompt):
            image = image_path(args.image_dir, prompt, sample)
            if not image.exists():
                raise FileNotFoundError(image)
            tasks.append((prompt, sample, image))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    rows = {}
    if results_path.exists():
        rows = {r["key"]: r for r in map(json.loads, results_path.read_text(encoding="utf-8").splitlines())}
    pending = [t for t in tasks if f"{t[0].group}/{t[0].item_id}/{t[1]}" not in rows]
    print(f"[tiif] {len(tasks)} images, {len(pending)} to judge", flush=True)
    failures = []
    if pending:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"set ${args.api_key_env} to the API key of the judge endpoint")
        with ThreadPoolExecutor(args.max_workers) as pool:
            futures = [pool.submit(judge, args, api_key, templates, *task) for task in pending]
            for n, future in enumerate(as_completed(futures), 1):
                try:
                    row = future.result()
                    rows[row["key"]] = row
                except RuntimeError as exc:
                    failures.append(str(exc))
                if n % 50 == 0 or n == len(futures):
                    write_jsonl(results_path, rows.values())
                    print(f"[tiif] {n}/{len(futures)} judged, {len(failures)} failed", flush=True)
    if len(failures) > args.max_failed:
        raise RuntimeError(f"{len(failures)} images have no usable judgment (tolerance {args.max_failed}); rerun to retry them")

    judged = [rows[key] for key in (f"{p.group}/{p.item_id}/{s}" for p, s, _ in tasks) if key in rows]
    groups: dict[str, list[bool]] = {}
    for row in judged:
        groups.setdefault(row["group"], []).extend(a == b for a, b in zip(row["answers"], row["reference"]))
    summary = {
        "benchmark": "tiif",
        "overall": 100 * mean(mean(values) for values in groups.values()),
        "groups": {group: 100 * mean(values) for group, values in sorted(groups.items())},
        "num_images": len(judged),
        "failed_images": len(tasks) - len(judged),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
