"""
Upload CodeRL APPS rollout dataset to HuggingFace.

Uploads per-problem files from data/APPS/train/:
  - gen_solutions_round1.json      (SFT rollout code + unit test results)
  - gen_solutions_critic_scores_round1.pkl  (round1 critic scores for RL training)
  - baseline_solutions_round1.json (SFT greedy baseline for relative_returns)

Usage:
    python scripts/upload_dataset_hf.py
    python scripts/upload_dataset_hf.py --dry-run   # check counts only
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo

REPO_ID = "lck0328/CodeRL-APPS-rollouts"
LOCAL_DIR = "data/APPS/train"
UPLOAD_PATTERNS = [
    "**/gen_solutions_round1.json",
    "**/gen_solutions_critic_scores_round1.pkl",
    "**/baseline_solutions_round1.json",
]


def dry_run(local_dir, patterns):
    import fnmatch
    counts = {p: 0 for p in patterns}
    for problem in sorted(os.listdir(local_dir)):
        problem_dir = os.path.join(local_dir, problem)
        if not os.path.isdir(problem_dir):
            continue
        for fname in os.listdir(problem_dir):
            fpath = os.path.join(problem, fname)
            for pat in patterns:
                glob_pat = pat.replace("**/", "")
                if fnmatch.fnmatch(fname, glob_pat):
                    counts[pat] += 1
    print("Files to upload:")
    for pat, count in counts.items():
        print(f"  {pat.replace('**/', ''):<45} {count} files")
    total = sum(counts.values())
    print(f"\nTotal: {total} files across {len(os.listdir(local_dir))} problem dirs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Count files without uploading")
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()

    if args.dry_run:
        dry_run(LOCAL_DIR, UPLOAD_PATTERNS)
        return

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        print(f"Repo ready: https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        print(f"Warning creating repo: {e}")

    print(f"Uploading from {LOCAL_DIR}/ ...")
    print("Patterns:", UPLOAD_PATTERNS)
    print("This may take a while (~1.2 GB total).\n")

    api.upload_folder(
        folder_path=LOCAL_DIR,
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=UPLOAD_PATTERNS,
        commit_message="Add SFT rollouts, round1 critic scores, and baseline solutions",
    )

    print(f"\nDone. View at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
