# scripts/check_unit_test_results.py
import pickle as pkl
import glob
import os
import json
from collections import Counter

codes_dir = "outputs/codes/"
results_dir = "outputs/test_results_SFT_actor_sampling/"

# Find all generated code files
code_files = glob.glob(codes_dir + "*.json")
code_ids = set(int(os.path.basename(f).replace('.json', '')) for f in code_files)

# Find all result pkl files
pkl_files = glob.glob(results_dir + "*.pkl")
pkl_ids = set(int(os.path.basename(f).replace('.pkl', '')) for f in pkl_files)

# Missing: has code but no pkl
missing = sorted(code_ids - pkl_ids)

# Check each pkl for validity
empty = []
broken = []
good = []
outcome_counter = Counter()

for pkl_file in sorted(pkl_files):
    try:
        data = pkl.load(open(pkl_file, 'rb'))
        idx = list(data.keys())[0]
        results = data[idx]['results']
        if len(results) == 0:
            empty.append(idx)
        else:
            good.append(idx)
            for r in results:
                if r == [-2]:
                    outcome_counter['compile_error'] += 1
                elif r == [-1]:
                    outcome_counter['runtime_error'] += 1
                elif r == [False]:
                    outcome_counter['failed_test'] += 1
                elif all(x is True for x in r):
                    outcome_counter['passed'] += 1
                else:
                    outcome_counter['partial'] += 1
    except Exception as e:
        broken.append(pkl_file)

print(f"=== Unit Test Results Summary ===")
print(f"Total code files     : {len(code_ids)}")
print(f"Total pkl files      : {len(pkl_ids)}")
print(f"Missing (no pkl)     : {len(missing)}")
print(f"Empty pkl            : {len(empty)}")
print(f"Broken pkl           : {len(broken)}")
print(f"Good pkl             : {len(good)}")
print(f"\n=== Outcome Distribution ===")
for k, v in sorted(outcome_counter.items()):
    print(f"  {k:20s}: {v}")

if missing:
    print(f"\n=== Missing IDs (first 20) ===")
    print(missing[:20])
if empty:
    print(f"\n=== Empty IDs (first 20) ===")
    print(empty[:20])
if broken:
    print(f"\n=== Broken files (first 20) ===")
    print(broken[:20])