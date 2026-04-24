
#!/usr/bin/env python3
"""
eval_results.py — Evaluate CodeRL unit-test pkl outputs.

Usage:
    python eval_results.py <unit_test_pkl_path>

Output:
    Prints a summary table and saves it to <unit_test_pkl_path>/eval_summary.txt
"""

import sys
import pickle as pkl
import glob
import os
import numpy as np
from math import comb
from datetime import datetime


# ─── helpers ──────────────────────────────────────────────────────────────────

def solution_passed(result_list):
    """A solution passes only if every test case returned True."""
    return all(r is True or r is True for r in result_list)

def classify_solution(result_list):
    """
    Classify a single solution's result list into one of four categories.
    Priority: CompileError > RuntimeError > FailedTest > PassedTest
    """
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


# ─── main ─────────────────────────────────────────────────────────────────────

def compute_metrics(results_dir):
    pkl_files = sorted(glob.glob(os.path.join(results_dir, "*.pkl")))
    if not pkl_files:
        print(f"No .pkl files found in: {results_dir}")
        sys.exit(1)

    n_problems = len(pkl_files)

    # per-problem accumulators
    pass1_list, pass5_list = [], []

    # solution-level outcome counts (across all problems × all samples)
    total_solutions = 0
    outcome_counts = {"CompileError": 0, "RuntimeError": 0, "FailedTest": 0, "PassedTest": 0}

    # problem-level outcome counts (dominant outcome per problem)
    problem_outcome_counts = {"CompileError": 0, "RuntimeError": 0, "FailedTest": 0, "PassedTest": 0}

    # raw signal counts (individual test-case verdicts)
    raw_counts = {"True": 0, "False": 0, "-1": 0, "-2": 0}

    # partial-pass tracking: problems where ≥1 but <all solutions pass
    n_any_pass = 0    # ≥1 solution passed
    n_all_pass = 0    # all solutions passed
    n_none_pass = 0   # zero solutions passed

    skipped = 0

    for fpath in pkl_files:
        with open(fpath, "rb") as f:
            data = pkl.load(f)
        idx = list(data.keys())[0]
        results = data[idx]["results"]   # list[list]

        n = len(results)
        if n == 0:
            skipped += 1
            continue

        total_solutions += n
        c = 0  # number of passing solutions for this problem
        sol_outcomes = []

        for result_list in results:
            # raw verdicts
            for r in result_list:
                if r is True:
                    raw_counts["True"] += 1
                elif r is False:
                    raw_counts["False"] += 1
                elif r == -1:
                    raw_counts["-1"] += 1
                elif r == -2:
                    raw_counts["-2"] += 1

            # solution-level outcome
            outcome = classify_solution(result_list)
            outcome_counts[outcome] += 1
            sol_outcomes.append(outcome)

            if solution_passed(result_list):
                c += 1

        # pass@k
        pass1_list.append(pass_at_k(n, c, 1))
        if n >= 5:
            pass5_list.append(pass_at_k(n, c, 5))

        # partial-pass breakdown
        if c == 0:
            n_none_pass += 1
        elif c == n:
            n_all_pass += 1
        else:
            n_any_pass += 1

        # dominant problem-level outcome (worst outcome wins, same priority order)
        dominant = classify_solution([v for rl in results for v in rl])
        problem_outcome_counts[dominant] += 1

    evaluated = n_problems - skipped
    pass1  = np.mean(pass1_list)  * 100 if pass1_list  else 0.0
    pass5  = np.mean(pass5_list)  * 100 if pass5_list  else 0.0

    # ── format report ──────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append(f"  CodeRL Eval Summary")
    lines.append(f"  Path   : {os.path.abspath(results_dir)}")
    lines.append(f"  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    lines.append("\n── Pass@k ──────────────────────────────────────────────────")
    lines.append(f"  Problems evaluated : {evaluated}  (skipped: {skipped})")
    lines.append(f"  Samples per problem: {total_solutions // evaluated if evaluated else 0}  (n={total_solutions} total)")
    lines.append(f"  pass@1             : {pass1:.2f}%")
    lines.append(f"  pass@5             : {pass5:.2f}%")

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
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    # ── save ───────────────────────────────────────────────────────────────────
    out_path = os.path.join(results_dir, "eval_summary.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval_results.py <unit_test_pkl_path>")
        sys.exit(1)
    compute_metrics(sys.argv[1])
    