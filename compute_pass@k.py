import pickle as pkl
import glob
import numpy as np
from math import comb

def solution_passed(result_list):
    """A solution passes only if every test case returned True."""
    return all(r is True or r == True for r in result_list)

def pass_at_k(n, c, k):
    """
    Unbiased HumanEval estimator.
    n = total samples per problem, 
    c = number that passed all tests, 
    k = k value.
    """
    
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def compute_metrics(results_dir, label):
    pkl_files = sorted(glob.glob(f"{results_dir}/*.pkl"))
    print(f"\n{label}: found {len(pkl_files)} pkl files")

    pass1_list, pass5_list = [], []
    skipped = 0

    for fpath in pkl_files:
        with open(fpath, 'rb') as f:
            data = pkl.load(f)
        idx = list(data.keys())[0]
        results = data[idx]['results']

        n = len(results)
        if n == 0:
            skipped += 1
            continue

        c = sum(1 for r in results if solution_passed(r))

        if n >= 1:
            pass1_list.append(pass_at_k(n, c, 1))
        if n >= 5:
            pass5_list.append(pass_at_k(n, c, 5))

    pass1 = np.mean(pass1_list) * 100 if pass1_list else 0.0
    pass5 = np.mean(pass5_list) * 100 if pass5_list else 0.0

    print(f"  Problems evaluated : {len(pkl_files) - skipped}")
    print(f"  pass@1             : {pass1:.2f}%")
    print(f"  pass@5             : {pass5:.2f}%")
    return pass1, pass5

sft_p1, sft_p5 = compute_metrics("outputs/results_sft", "SFT")
rl_p1,  rl_p5  = compute_metrics("outputs/results_rl",  "RL")

output = f"""
=== CodeRL Reproduction Results (Introductory, 500 problems, n=10) ===

Model   | pass@1  | pass@5
--------|---------|--------
SFT     | {sft_p1:6.2f}% | {sft_p5:6.2f}%
RL      | {rl_p1:6.2f}% | {rl_p5:6.2f}%
"""

print(output)
with open("results_summary.txt", "w") as f:
    f.write(output)
print("Saved to results_summary.txt")