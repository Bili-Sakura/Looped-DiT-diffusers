"""SpatialGenEval scoring: the official 5-rollout majority vote with Qwen2.5-VL-72B.

For every image the judge answers the prompt's 10 multiple-choice questions five
times at temperature 1; a question counts as correct when the right option wins
at least 4 of the 5 rollouts. Needs `vllm` and `qwen-vl-utils`.

    python -m looped_dit.eval.spatial_geneval --data eval_assets/spatial_geneval/SpatialGenEval_T2I_Prompts.jsonl \
        --image-dir OUT/spatial_geneval_images --out-dir OUT/spatial_geneval_results
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from .benchmarks import image_path, load_prompts
from .judge import Request, VLLMJudge, strip_thinking, write_jsonl

SYSTEM_PROMPT = "You are a professional image critic. Answer only from visible evidence in the image."
PROMPT_TEMPLATE = """### Task Description:
Carefully examine the image and answer the following 10 multiple-choice questions.
Only rely on the image. Do not infer answers from the text-to-image prompt or external knowledge.

### Multiple-Choice Questions:
{questions}

### Instructions:
1. Answer the 10 questions on exactly 10 separate lines and preserve their order.
2. Begin each line with exactly one option letter followed by a colon, such as `A:` or `E:`.
3. Give a brief reason after the option on the same line.
4. Select `E: None` when the image does not provide enough visual evidence.
"""
ANSWER_LINE = re.compile(r"^\s*(?:\d+\s*[.)-]\s*)?(?:answer\s*\d*\s*[:=-]\s*)?\**([A-E])\**\s*[:.)-]", re.IGNORECASE)


def parse_options(response: str, expected: int = 10) -> list[str]:
    options = [m.group(1).upper() for line in strip_thinking(response).splitlines() if (m := ANSWER_LINE.match(line))]
    if len(options) != expected:
        raise ValueError(f"expected {expected} answers, parsed {len(options)}")
    return options


def main() -> None:
    parser = argparse.ArgumentParser(description="Score SpatialGenEval images with a VLM judge.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--majority", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    args = parser.parse_args()

    requests = []
    for prompt in load_prompts("spatial_geneval", args.data):
        for sample in range(args.samples_per_prompt):
            image = image_path(args.image_dir, prompt, sample)
            if not image.exists():
                raise FileNotFoundError(image)
            questions = "\n".join(str(q).strip() for q in prompt.metadata["questions"])
            text = PROMPT_TEMPLATE.format(questions=questions)
            requests.append(Request(prompt, sample, "multiple_choice", 0, image, SYSTEM_PROMPT, text, 1))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    rows = {}
    if results_path.exists():
        rows = {r["key"]: r for r in map(json.loads, results_path.read_text(encoding="utf-8").splitlines())}
    pending = {r.key: r for r in requests if r.key not in rows}
    valid: dict[str, list[list[str]]] = {key: [] for key in pending}
    if pending:
        judge = VLLMJudge(args.judge_model, args.tensor_parallel_size, args.max_model_len)
        for _ in range(args.retries):  # draw rollouts until every image has enough parseable ones
            keys = list(pending)
            for start in range(0, len(keys), args.batch_size):
                batch = [pending[k] for k in keys[start : start + args.batch_size]]
                outputs = judge.generate(batch, args.max_new_tokens, args.temperature, n=args.rollouts)
                for request, responses in zip(batch, outputs):
                    for response in responses:
                        try:
                            valid[request.key].append(parse_options(response))
                        except ValueError:
                            continue
                    if len(valid[request.key]) < args.rollouts:
                        continue
                    rollouts = valid[request.key][: args.rollouts]
                    answers = [str(a).upper() for a in request.prompt.metadata["answers"]]
                    correct = [
                        Counter(r[i] for r in rollouts)[answer] >= args.majority for i, answer in enumerate(answers)
                    ]
                    rows[request.key] = {"key": request.key, "item_id": request.prompt.item_id,
                                         "scene": request.prompt.group, "correct": correct, "rollouts": rollouts}
                    del pending[request.key]
                write_jsonl(results_path, rows.values())
            if not pending:
                break
    if pending:
        raise RuntimeError(f"{len(pending)} images did not get {args.rollouts} parseable rollouts")

    scored = [rows[r.key] for r in requests]
    summary = {
        "benchmark": "spatial_geneval",
        "overall": 100 * mean(c for row in scored for c in row["correct"]),
        "scenes": {s: 100 * mean(c for row in scored if row["scene"] == s for c in row["correct"])
                   for s in sorted({row["scene"] for row in scored})},
        "num_images": len(scored),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
