#!/usr/bin/env python3
"""
make_baseline_solutions.py — Convert greedy baseline codes + unit test results
into baseline_solutions.json per problem in the APPS train directory.

Usage:
    python scripts/make_baseline_solutions.py \
        --codes_dir outputs/sampled_code/codes_baseline/ \
        --results_dir outputs/unit_test_score/test_results_baseline/ \
        --train_path data/APPS/train/
"""

import argparse
import glob
import json
import os
import pickle


def reduce_result(result_list):
    """Reduce a list of per-test-case results to a single worst-case value."""
    if any(r == -2 for r in result_list):
        return -2
    if any(r == -1 for r in result_list):
        return -1
    if any(r is False for r in result_list):
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes_dir", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--train_path", required=True)
    args = parser.parse_args()

    # Build index → problem_dir mapping (same as generate.py)
    problem_dirs = sorted(glob.glob(os.path.join(args.train_path, "*")))
    idx_to_dir = {i: d for i, d in enumerate(problem_dirs)}

    code_files = sorted(glob.glob(os.path.join(args.codes_dir, "*.json")),
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    written, skipped = 0, 0

    for code_file in code_files:
        idx = int(os.path.splitext(os.path.basename(code_file))[0])
        problem_dir = idx_to_dir.get(idx)
        if problem_dir is None:
            skipped += 1
            continue

        # Load generated code (single greedy sample)
        with open(code_file) as f:
            cdata = json.load(f)
        prob_key = list(cdata.keys())[0]
        codes = cdata[prob_key]["code"]
        if not codes:
            skipped += 1
            continue
        code = codes[0]

        # Load unit test result
        result_file = os.path.join(args.results_dir, f"{idx}.pkl")
        if not os.path.exists(result_file):
            skipped += 1
            continue
        with open(result_file, "rb") as f:
            rdata = pickle.load(f)
        results = list(rdata.values())[0]["results"]
        if not results:
            skipped += 1
            continue

        # Reduce per-test-case results to single value
        result = reduce_result(results[0])

        # Write baseline_solutions.json
        out = [{"code": code, "result": result}]
        out_path = os.path.join(problem_dir, "baseline_solutions.json")
        with open(out_path, "w") as f:
            json.dump(out, f)
        written += 1

    print(f"Done. Written: {written}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
