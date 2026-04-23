import json
import os
import glob
import pickle as pkl
from tqdm import tqdm

codes_dir = "outputs/codes/"
results_dir = "outputs/test_results_SFT_actor_sampling/"
train_dir = "data/APPS/train/"

code_files = sorted(glob.glob(codes_dir + "*.json"))
print(f"Found {len(code_files)} code files")

skipped = 0
converted = 0

for code_file in tqdm(code_files):
    problem_id = int(os.path.basename(code_file).replace('.json', ''))
    pkl_file = os.path.join(results_dir, f"{problem_id}.pkl")

    # Skip if no pkl (missing input_output.json problems)
    if not os.path.exists(pkl_file):
        skipped += 1
        continue

    # Load codes
    with open(code_file) as f:
        code_data = json.load(f)
    codes = code_data[str(problem_id)]['code']

    # Load test results
    try:
        with open(pkl_file, 'rb') as f:
            pkl_data = pkl.load(f)
        idx = list(pkl_data.keys())[0]
        results = pkl_data[idx]['results']
    except Exception:
        skipped += 1
        continue

    # Merge code + result
    # result per solution: use first element of result list as the result signal
    gen_solutions = []
    for i, code in enumerate(codes):
        if i < len(results):
            r = results[i]
            # Collapse result list to single value
            if r == [-2]:
                result = -2
            elif r == [-1]:
                result = -1
            elif all(x is True for x in r):
                result = True
            else:
                result = False
        else:
            result = -2  # fallback
        gen_solutions.append({'code': code, 'result': result})

    # Write to problem folder
    out_path = os.path.join(train_dir, str(problem_id).zfill(4), "gen_solutions.json")
    with open(out_path, 'w') as f:
        json.dump(gen_solutions, f)
    converted += 1

print(f"\nConverted : {converted}")
print(f"Skipped   : {skipped}")
print(f"Done. gen_solutions.json written to each problem folder.")