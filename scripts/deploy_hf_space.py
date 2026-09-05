"""Publish the app to a Hugging Face Space (Streamlit SDK).

Usage (needs a write token in HF_TOKEN or a prior `hf auth login`):

    python scripts/deploy_hf_space.py --space <user>/auto-experimenter

Uploads the code, prompts, data, saved results and the Space README (with the
front matter Hugging Face needs), then prints the URL. The ANTHROPIC_API_KEY
is NOT uploaded; set it as a Space secret in the Space settings if you want the
live Claude planner and narrator.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
IGNORE = [
    ".git/*", ".git*", "__pycache__/*", "*/__pycache__/*", "*.pyc", ".pytest_cache/*", ".env",
    ".claude/*", ".streamlit/secrets.toml", "results/live/*", "results/latest/*", "docs/*",
    "tests/*", "scripts/*", "submission/*", "README.md", "pyproject.toml", "requirements-dev.txt",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="repo id, e.g. aimanelasad/auto-experimenter")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=args.space, repo_type="space", space_sdk="streamlit", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=args.space,
        repo_type="space",
        ignore_patterns=IGNORE,
        commit_message="Deploy auto-experimenter",
    )
    api.upload_file(
        path_or_fileobj=str(ROOT / "deploy" / "README_space.md"),
        path_in_repo="README.md",
        repo_id=args.space,
        repo_type="space",
        commit_message="Space README",
    )
    print(f"Deployed: https://huggingface.co/spaces/{args.space}")
    print("Optional: add ANTHROPIC_API_KEY under Settings > Variables and secrets to enable the live Claude planner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
