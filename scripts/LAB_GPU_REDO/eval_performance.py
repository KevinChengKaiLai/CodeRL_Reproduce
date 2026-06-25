
#!/usr/bin/env python3
"""
eval_performance.py — Evaluate CodeRL unit-test pkl outputs.

Usage:
    python eval_performance.py <unit_test_pkl_path> [codes_dir] [--k 1 5 10 20]

    codes_dir (optional): path to the generated codes JSON directory.
                          When provided, adds code length and uniqueness metrics.
    --k (optional): space-separated list of k values for pass@k.
                    Defaults to all of {1, 5, 10, n} where n is the detected
                    number of samples per problem (duplicates removed, sorted).

Output:
    Prints a summary table and saves it to <unit_test_pkl_path>/eval_summary.txt
"""

import sys
import argparse
import pickle as pkl
import glob
import os
import json
import numpy as np
from math import comb
from datetime import datetime
from collections import defaultdict


# ─── helpers ──────────────────────────────────────────────────────────────────

def solution_passed(result_list):
    return all(r is True for r in result_list)

def classify_solution(result_list):
    if any(r == -2 for r in result_list):
        return "CompileError"
    if any(r == -1 for r in result_list):
        return "RuntimeError"
    if all(r is True for r in result_list):
        return "PassedTest"
    return "FailedTest"

def pass_at_k(n, c, k):
    """Unbiased HumanEval estimator. n=total, c=passed, k=k."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def partial_pass_rate(result_list):
    """Fraction of individual test cases that returned True."""
    if not result_list:
        return 0.0
    return sum(1 for r in result_list if r is True) / len(result_list)

def uniqueness_rate(codes):
    """Fraction of codes that are unique within the list."""
    if not codes:
        return 0.0
    return len(set(codes)) / len(codes)


# ─── main ─────────────────────────────────────────────────────────────────────

def compute_metrics(results_dir, codes_dir=None, k_values=None):
    pkl_files = sorted(glob.glob(os.path.join(results_dir, "*.pkl")))
    if not pkl_files:
        print(f"No .pkl files found in: {results_dir}")
        sys.exit(1)

    n_problems = len(pkl_files)

    # Detect n (samples per problem) from first non-empty pkl
    detected_n = None
    for fpath in pkl_files:
        with open(fpath, "rb") as f:
            data = pkl.load(f)
        idx = list(data.keys())[0]
        n = len(data[idx]["results"])
        if n > 0:
            detected_n = n
            break

    if k_values is None:
        # Auto: 1, 5, 10, and detected_n (deduped, sorted, clamped to detected_n)
        defaults = {1, 5, 10}
        if detected_n is not None:
            defaults.add(detected_n)
        k_values = sorted(k for k in defaults if detected_n is None or k <= detected_n)
    else:
        if detected_n is not None:
            k_values = sorted(k for k in k_values if k <= detected_n)

    # per-k accumulators: k -> list of per-problem pass@k values
    pass_at_k_lists = defaultdict(list)

    total_solutions = 0
    outcome_counts = {"CompileError": 0, "RuntimeError": 0, "FailedTest": 0, "PassedTest": 0}
    problem_outcome_counts = {"CompileError": 0, "RuntimeError": 0, "FailedTest": 0, "PassedTest": 0}
    raw_counts = {"True": 0, "False": 0, "-1": 0, "-2": 0}

    n_any_pass = 0
    n_all_pass = 0
    n_none_pass = 0

    partial_rates = []
    skipped = 0

    code_lengths = []
    uniqueness_rates = []

    for fpath in pkl_files:
        with open(fpath, "rb") as f:
            data = pkl.load(f)
        idx = list(data.keys())[0]
        results = data[idx]["results"]

        n = len(results)
        if n == 0:
            skipped += 1
            continue

        total_solutions += n
        c = 0

        for result_list in results:
            for r in result_list:
                if r is True:
                    raw_counts["True"] += 1
                elif r is False:
                    raw_counts["False"] += 1
                elif r == -1:
                    raw_counts["-1"] += 1
                elif r == -2:
                    raw_counts["-2"] += 1

            outcome = classify_solution(result_list)
            outcome_counts[outcome] += 1

            if solution_passed(result_list):
                c += 1
            else:
                partial_rates.append(partial_pass_rate(result_list))

        for k in k_values:
            if n >= k:
                pass_at_k_lists[k].append(pass_at_k(n, c, k))

        if c == 0:
            n_none_pass += 1
        elif c == n:
            n_all_pass += 1
        else:
            n_any_pass += 1

        dominant = classify_solution([v for rl in results for v in rl])
        problem_outcome_counts[dominant] += 1

        if codes_dir is not None:
            code_file = os.path.join(codes_dir, f"{idx}.json")
            if os.path.exists(code_file):
                with open(code_file) as cf:
                    cdata = json.load(cf)
                codes = cdata.get(str(idx), cdata.get(idx, {})).get("code", [])
                if codes:
                    code_lengths.extend(len(c) for c in codes)
                    uniqueness_rates.append(uniqueness_rate(codes))

    evaluated = n_problems - skipped
    avg_partial = np.mean(partial_rates) * 100 if partial_rates else 0.0

    # ── format report ──────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append(f"  CodeRL Eval Summary")
    lines.append(f"  Path   : {os.path.abspath(results_dir)}")
    lines.append(f"  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    lines.append("\n── Pass@k ──────────────────────────────────────────────────")
    lines.append(f"  Problems evaluated : {evaluated}  (skipped: {skipped})")
    n_per_prob = total_solutions // evaluated if evaluated else 0
    lines.append(f"  Samples per problem: {n_per_prob}  (n={total_solutions} total)")
    for k in k_values:
        vals = pass_at_k_lists[k]
        pct = np.mean(vals) * 100 if vals else 0.0
        n_eligible = len(vals)
        lines.append(f"  pass@{k:<3}            : {pct:.2f}%  ({n_eligible} problems with n>={k})")

    lines.append("\n── Partial Test Pass Rate ───────────────────────────────────")
    lines.append(f"  Avg test cases passed (failing solutions): {avg_partial:.2f}%")
    lines.append(f"  (How close failing solutions are to passing)")

    lines.append("\n── Solution-level Outcomes (% of all solutions) ────────────")
    for label in ["PassedTest", "FailedTest", "RuntimeError", "CompileError"]:
        count = outcome_counts[label]
        pct = count / total_solutions * 100 if total_solutions else 0
        lines.append(f"  {label:<14}: {count:>6}  ({pct:5.1f}%)")

    lines.append("\n── Problem-level Dominant Outcome (% of problems) ──────────")
    for label in ["PassedTest", "FailedTest", "RuntimeError", "CompileError"]:
        count = problem_outcome_counts[label]
        pct = count / evaluated * 100 if evaluated else 0
        lines.append(f"  {label:<14}: {count:>6}  ({pct:5.1f}%)")

    lines.append("\n── Raw Test-Case Verdicts ───────────────────────────────────")
    raw_total = sum(raw_counts.values())
    lines.append(f"  True  (PassedTest)  : {raw_counts['True']:>7}  ({raw_counts['True']/raw_total*100:5.1f}%)")
    lines.append(f"  False (FailedTest)  : {raw_counts['False']:>7}  ({raw_counts['False']/raw_total*100:5.1f}%)")
    lines.append(f"  -1    (RuntimeError): {raw_counts['-1']:>7}  ({raw_counts['-1']/raw_total*100:5.1f}%)")
    lines.append(f"  -2    (CompileError): {raw_counts['-2']:>7}  ({raw_counts['-2']/raw_total*100:5.1f}%)")

    lines.append("\n── Problem Solve Rate Breakdown ────────────────────────────")
    lines.append(f"  Problems with ALL solutions passing : {n_all_pass:>5}  ({n_all_pass/evaluated*100:5.1f}%)")
    lines.append(f"  Problems with SOME solutions passing: {n_any_pass:>5}  ({n_any_pass/evaluated*100:5.1f}%)")
    lines.append(f"  Problems with NO  solution passing  : {n_none_pass:>5}  ({n_none_pass/evaluated*100:5.1f}%)")

    if code_lengths:
        lines.append("\n── Generated Code Length (chars) ───────────────────────────")
        lines.append(f"  Mean   : {np.mean(code_lengths):>8.1f}")
        lines.append(f"  Median : {np.median(code_lengths):>8.1f}")
        lines.append(f"  Std    : {np.std(code_lengths):>8.1f}")
        lines.append(f"  Min    : {np.min(code_lengths):>8}    Max: {np.max(code_lengths)}")

        lines.append("\n── Sample Uniqueness (diversity) ───────────────────────────")
        lines.append(f"  Mean uniqueness rate : {np.mean(uniqueness_rates)*100:.1f}%")
        lines.append(f"  (% of distinct solutions per problem, avg across problems)")
        lines.append(f"  Problems with all-identical samples: {sum(1 for r in uniqueness_rates if r < 1/len(uniqueness_rates)+1e-9):>5}")

    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    out_path = os.path.join(results_dir, "eval_summary.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {out_path}")

    import json as _json
    json_out = {
        "pass_at_k": {str(k): round(float(np.mean(pass_at_k_lists[k])) * 100, 4)
                      for k in k_values if pass_at_k_lists[k]},
        "outcome_pcts": {
            label: round(outcome_counts[label] / total_solutions * 100, 2)
            for label in ["PassedTest", "FailedTest", "RuntimeError", "CompileError"]
        } if total_solutions else {},
        "uniqueness_mean_pct": round(float(np.mean(uniqueness_rates)) * 100, 2) if uniqueness_rates else None,
        "n_problems_evaluated": evaluated,
        "n_samples_per_problem": n_per_prob,
        "results_dir": os.path.abspath(results_dir),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path = os.path.join(results_dir, "eval_summary.json")
    with open(json_path, "w") as f:
        _json.dump(json_out, f, indent=2)
    print(f"Saved JSON to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CodeRL unit-test pkl outputs.")
    parser.add_argument("results_dir", help="Directory containing .pkl unit test result files")
    parser.add_argument("codes_dir", nargs="?", default=None,
                        help="Optional directory of generated code JSON files (adds length/uniqueness metrics)")
    parser.add_argument("--k", nargs="+", type=int, default=None,
                        help="k values for pass@k (default: auto-detect from {1,5,10,n})")
    args = parser.parse_args()
    compute_metrics(args.results_dir, args.codes_dir, args.k)
