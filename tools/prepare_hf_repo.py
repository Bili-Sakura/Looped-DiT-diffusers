"""Refresh the Hugging Face repo layout (code and configs, no weights).

    python tools/prepare_hf_repo.py
    python tools/prepare_hf_repo.py --root Looped-DiT-diffusers

Writes `Looped-DiT-B-32/`, `Looped-DiT-B-16/`, and `Looped-DiT-L-16/` under the root.
Each variant can then be loaded by diffusers once a checkpoint has been converted into it:

    python tools/convert_to_diffusers.py --checkpoint checkpoints/looped-dit-b16.pt \\
        --output-dir Looped-DiT-diffusers/Looped-DiT-B-16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looped_dit.checkpoint import HF_REPO_NAME, prepare_hf_repo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=HF_REPO_NAME, help="Hugging Face repo directory to fill")
    args = parser.parse_args()
    root = prepare_hf_repo(args.root)
    print(f"wrote {root}")


if __name__ == "__main__":
    main()
