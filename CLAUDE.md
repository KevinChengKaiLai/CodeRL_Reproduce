# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reproduction of **CodeRL** (NeurIPS 2022), an actor-critic RL framework for code generation using CodeT5. An actor (CodeT5 language model) generates Python programs; a critic predicts unit test outcomes (compile error, runtime error, failed tests, passed) and provides reward signals.

**Current status:** Round 5 gradient filter ablation complete (2026-06-23). 5A (`--positive_only`) is the only stable configuration; 5B/5C/5D all collapsed. 5E (gnorm=1.0) pending.

See `round5_rl.md` for full Round 5 results and analysis.
See `round2_rl.md` for Rounds 2–4 session notes and post-mortems.
See `round1_rl.md` for full Round 1 analysis.

### Round 5 Experiment Results (gradient filter ablation)
All experiments: warmstart=ckpt-3000, LR=5e-6, data=round1 (SFT rollouts), relative_returns.
Single variable: gradient filter.

| Exp | Gradient filter | gnorm | Data | Status |
|---|---|---|---|---|
| **5A** | `--positive_only` | 0.1 | round1 | done — stable, pass@10=1.00%, CE%≈3% |
| **5B** | none (all samples) | 0.1 | round1 | done — collapsed (CE% 46→67%) |
| **5C** | `--negative_only` | 0.1 | round1 | done — collapsed (CE%=100% from ckpt-1000) |
| **5D** | pos+neg (weak baseline) | 0.1 | round1 | done — collapsed (CE% 60–71%) |
| **5E** | `--positive_only` | 1.0 | round1 | pending (gnorm ablation vs 5A) |

**Full training pipeline (one round):**
1. SFT — supervised fine-tuning on ground-truth APPS solutions
2. Actor sampling — stochastic decode (temp=0.6, n=20) from actor for all 5000 train problems
3. Unit tests — run hidden tests on generated programs, get pass/fail signals
4. Critic training — train CodeT5-small to predict 4-class test outcomes
5. Critic scoring — score all generated programs with trained critic
6. Baseline generation — greedy decode (temp=0) from actor for relative-returns reward
7. RL training — policy gradient with actor + critic feedback + relative returns

--/

## Setup

```bash
pip install -r requirements.txt
cd transformers && pip install -e . && cd ..  # REQUIRED: local fork with tuning_mode support
```

The `transformers/` directory is a **local fork of HuggingFace Transformers v4.16.1** with modifications for `tuning_mode` (SFT/critic/RL heads in `transformers/src/transformers/models/t5/`). Must be installed instead of the standard package.

---

## Models

| Path | What |
|---|---|
| `models/codet5-base-tokenizer/` | RobertaTokenizer — always use local path (no network) |
| `models/codet5-base/` | Pre-trained CodeT5-base weights |
| `models/codet5-small/` | Pre-trained CodeT5-small weights |
| `models/sft_actor/` | SFT fine-tuned CodeT5-large (~10 epochs on APPS train) |
| `models/rl_actor_best_ckpt3000/` | **Best RL checkpoint** (round 1, step 3000, pass@10=1.00%) |
| `exps/codet5-large_rl_bs4x16_lr2e-05_newcritic_relreturns/` | Round 1 RL run (ckpts 5000–14000; 1–4k pruned) |
| `exps/codet5-small_critic_bs16x16_lr5e-05/checkpoint-8200` | Round 1 critic (trained on SFT rollouts) |
| `exps/codet5-small_critic_round2_bs32x8_lr2e-05/checkpoint-500` | Round 2 critic best ckpt (val_loss minimum at epoch 1; used for Phase F) |
| `exps/codet5-large_rl_round2_bs4x16_lr1e-05_relreturns/` | Round 2 RL run (collapsed at step 1000; all ckpts CE%=100%) |
| `exps/codet5-large_rl_round3_sft_lr1e-05_relreturns/` | Round 3 RL run (in progress — SFT start, round 1 data, LR=1e-5) |
| `models/rl_round3_best/` | Round 3 preserved checkpoints (watcher copies ckpts beating SFT baseline here) |

> **`train_configs.py` always prepends `exps/` to `--save_dir`.** Pass bare name: `--save_dir my_run` → saves to `exps/my_run/`.

---

## Data Layout

```
data/APPS/train/{0000..4999}/
    input_output.json               # hidden unit tests (absent for ~198 problems)
    solutions.json                  # ground-truth solutions (for SFT)
    gen_solutions.json              # generated solutions + test results (for critic/RL)
    gen_solutions_round1.json       # backup of round 1 gen_solutions
    gen_solutions_critic_scores.pkl # critic reward scores per solution
    gen_solutions_critic_scores_round1.pkl
    baseline_solutions.json         # greedy decode result (for relative returns)
    baseline_solutions_round1.json  # backup of round 1 baseline

data/APPS/test/{0000..4999}/       # eval split (500 intro problems used for eval)

outputs/sampled_code/
    codes/                          # SFT actor stochastic samples (round 1 training data)
    codes_baseline/                 # SFT actor greedy samples (round 1 baseline)
    codes_round2/                   # ckpt-3000 stochastic samples (round 2 training data)
    codes_baseline_round2/          # ckpt-3000 greedy samples (round 2 baseline)
    codes_relrl_ckpt_{N}/           # eval samples per round 1 checkpoint
    codes_sft/                      # SFT actor eval samples
    codes_round2_watch_ckpt_{N}/    # watcher eval samples per round 2 RL checkpoint (50 problems)
    codes_round3_watch_ckpt_{N}/    # watcher eval samples per round 3 RL checkpoint (50 problems)

outputs/unit_test_score/
    test_results_SFT_actor_sampling/  # unit test results for SFT training rollouts
    test_results_round2/              # unit test results for round 2 training rollouts
    test_results_baseline/            # unit test results for SFT greedy baseline
    test_results_baseline_round2/     # unit test results for ckpt-3000 greedy baseline
    test_results_relrl_ckpt_{N}/      # eval unit test results + eval_summary.txt per ckpt
    test_results_round2_watch_ckpt_{N}/  # watcher eval results per round 2 RL checkpoint
    test_results_round3_watch_ckpt_{N}/  # watcher eval results per round 3 RL checkpoint

outputs/archive/                    # old pre-round1 experiment outputs (kept for reference)

exps/                               # training checkpoints (gitignored)
logs/                               # all training/eval/pipeline logs
```

---

## Key Scripts

### Round 2 Pipeline (complete — A–F done; G collapsed)
```bash
# Individual phases (all complete):
bash scripts/round2_sample_actor.sh       # Phase A: sampling from rl_actor_best_ckpt3000 ✓
bash scripts/round2_unit_tests_train.sh  # Phase B: unit tests on round2 codes ✓
# Phase C: convert (inline in run_round2_pipeline.sh) ✓
bash scripts/round2_baseline.sh          # Phase D: greedy baseline from ckpt-3000 ✓
bash scripts/round2_train_critic.sh      # Phase E: fine-tune critic on round2 data ✓
bash scripts/round2_critic_scores.sh     # Phase F: generate critic scores (ckpt-500) ✓
bash scripts/round2_train_rl.sh          # Phase G: RL — COLLAPSED at step 1000 ✗
```

### Round 3 RL (running — SFT start, round 1 data, LR=1e-5)
```bash
# Training (GPU0, PID 3884625):
bash scripts/round3_train_rl.sh
# Uses --data-suffix _round1 → reads gen_solutions_critic_scores_round1.pkl + baseline_solutions_round1.json
# Save dir: exps/codet5-large_rl_round3_sft_lr1e-05_relreturns/
# 14,990 steps total (~54h); checkpoint every 1000 steps; save_total_limit=15 (keeps all)

# Live checkpoint eval watcher (GPU1, PID 3937358, polls every 5 min):
bash scripts/watch_rl_eval_round3.sh > logs/watch_rl_eval_round3.log 2>&1 &
# Samples 50 intro test problems (n=10), writes eval_summary.json per ckpt
# Preserves any ckpt beating SFT baseline (pass@10>0.60%, CE%<17.5%) → models/rl_round3_best/
```

**Why round 1 data (not round 2):** Grad norm analysis (`diagnose_grad_norm.py`) confirmed round 2 data causes 20× larger gradients for negative-reward batches. Root cause: ckpt-3000's stochastic samples have 0% pass rate (entropy collapsed) → 81% negative relative returns, only +0.4 max reward. Round 1 SFT data has 32% pass rate, 13.2% positive returns, reward range −2.0 to +2.0.

### Actor Sampling (training data generation)
```bash
# Both GPUs in parallel, 20 samples/problem, all 5000 train problems:
bash scripts/LAB_GPU_REDO/actor_sampling.sh
# Outputs: outputs/sampled_code/codes/  (round 1 SFT rollouts)
```

### Unit Tests on Training Data
```bash
bash scripts/LAB_GPU_REDO/run_unit_test_ActorSampling.sh
# Outputs: outputs/unit_test_score/test_results_SFT_actor_sampling/
```

### Convert codes + test results → gen_solutions.json
```bash
python convert_to_gen_solutions.py \
    --codes_dir outputs/sampled_code/codes/ \
    --results_dir outputs/unit_test_score/test_results_SFT_actor_sampling/ \
    --train_dir data/APPS/train/
# Writes data/APPS/train/{N}/gen_solutions.json for each problem
```

### Baseline Generation (for --relative_returns)
```bash
bash scripts/generate_baseline.sh
# Greedy decode (temp=0) + unit tests + writes baseline_solutions.json per problem
# Required before any RL training with --relative_returns
```

### Critic Training
```bash
# Train from scratch:
CUDA_VISIBLE_DEVICES=0 python train.py --model codet5-small --tuning_mode critic \
    --train_path data/APPS/train/ --save_dir codet5-small_critic_bs32x8_lr5e-05 \
    --epochs 10 --batch-size-per-replica 32 --grad-acc-steps 8 --lr 5e-5 --fp16

# Fine-tune from existing critic:
bash scripts/round2_train_critic.sh
```

### Generate Critic Scores
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_path exps/codet5-small_critic_bs16x16_lr5e-05/checkpoint-8200 \
    --tokenizer_path models/codet5-base-tokenizer \
    --test_path data/APPS/train/ --critic_scores --binary_prediction
# Writes data/APPS/train/{N}/gen_solutions_critic_scores.pkl
```

### RL Training
```bash
# Round 1 config (reference):
bash scripts/LAB_GPU_REDO/train_rl_actor.sh
# Round 2 (collapsed, reference only):
bash scripts/round2_train_rl.sh
# Round 3 (complete — SFT start, round 1 data):
bash scripts/round3_train_rl.sh
# Round 5 ablations (gradient filter, ckpt-3000 warmstart, round1 data):
bash scripts/round5a_train_rl.sh   # positive_only, gnorm=0.1
bash scripts/round5b_train_rl.sh   # no filter, gnorm=0.1 (done — collapsed)
bash scripts/round5c_train_rl.sh   # negative_only, gnorm=0.1
bash scripts/round5e_train_rl.sh   # positive_only, gnorm=1.0
```

### Evaluation
```bash
# Evaluate all relrl checkpoints (round 1):
bash scripts/eval_relrl_ckpts.sh > logs/eval_relrl.log 2>&1 &

# Parallel eval (ckpts 12000/13000/14000 simultaneously):
bash scripts/eval_relrl_parallel.sh > logs/eval_relrl_parallel.log 2>&1 &

# Pass@k from raw pkl (auto-detects n, reports pass@1/5/10/n):
python compute_pass@k.py
python scripts/LAB_GPU_REDO/eval_performance.py <results_dir> [codes_dir] [--k 1 5 10 20]
# Round 2 example (n=20 samples → also reports pass@20 automatically):
python scripts/LAB_GPU_REDO/eval_performance.py outputs/unit_test_score/test_results_round2_ckpt_X/ outputs/sampled_code/codes_round2_ckpt_X/
```

---

## RL Training Configuration (what works)

### Batch size for CodeT5-large RL
`codet5-large` (770M) with two forward passes per RL step requires ~22–23 GB on an RTX A5000 (24 GB).  
**Working config:** `--batch-size-per-replica 2 --grad-acc-steps 32` (effective batch = 64).  
`batch-size-per-replica 4` causes OOM. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### `--relative_returns` is mandatory
Without it, the reward signal has no baseline → model collapses to 99%+ compile errors within 7000 steps. Three failed runs confirmed this. Always include `--relative_returns` and ensure `baseline_solutions.json` exists per problem before training starts.

### `apps_dataset.py` guard for missing baseline
Problems without `input_output.json` have no `baseline_solutions.json`. The dataset loader guards against this (line ~162–166). If adding new problems, ensure the guard is present.

### Learning rate
- Round 1: LR=2e-5 → peaked at step 3000, then oscillating collapse
- Round 2: LR=1e-5, warmstarted from rl_actor_best_ckpt3000 → catastrophic collapse at step 1000 (CE%=100%). Root cause: ckpt-3000 was already at round 1's stability limit; a second RL pass from it collapsed immediately regardless of LR.
- Round 3: LR=1e-5, warmstart from `models/sft_actor/`, **round 1 data** (`--data-suffix _round1`) — running since 2026-05-25
- If collapse recurs at 1e-5, try 5e-6 or add KL penalty

### `save_total_limit`
Round 1 used 10 → best checkpoint (step 3000) was pruned mid-training. **Use 15+** to ensure the early peak is never lost. Round 2 uses `--save_total_limit 15`.

---

## Architecture

### Entry Points
- `train.py` — selects standard HF Trainer or `Trainer_RL` based on `tuning_mode`
- `generate.py` — sampling (stochastic or greedy) or critic score inference
- `test_one_solution.py` — execute unit tests for a single APPS problem

### Key Modules
- `datasets/apps_dataset.py` — APPS dataset loader; three modes: SFT (ground-truth), generated (synthetic programs for critic), RL (synthetic + critic hidden states + baseline)
- `datasets/utils.py` — `reindent_code()`, `get_error_type()` (maps -2/-1/False/True → classes 0–3), `get_reward_from_error_type()` (RL reward -1.0 to +1.0)
- `trainers/trainer_rl.py` — custom HF Trainer subclass implementing policy gradient RL loop
- `utils/testing_util.py` — `run_test()`: executes generated code against APPS unit tests
- `configs/train_configs.py` — argparse; note: always prepends `exps/` to `--save_dir`
- `convert_to_gen_solutions.py` — merges sampled codes + unit test pkl → `gen_solutions.json`
- `scripts/make_baseline_solutions.py` — merges greedy codes + pkl → `baseline_solutions.json`
- `scripts/LAB_GPU_REDO/eval_performance.py` — computes pass@k, uniqueness, error distribution; auto-detects n and reports pass@{1,5,10,n} by default; supports `--k` flag for custom k values; writes both `eval_summary.txt` (human-readable) and `eval_summary.json` (machine-readable) to results dir
- `scripts/watch_rl_eval.sh` / `scripts/watch_rl_eval_round3.sh` — live checkpoint watcher; polls every 5 min on GPU1; samples 50 intro test problems (n=10), runs unit tests, reads `eval_summary.json`, preserves ckpts beating SFT baseline
- `diagnose_grad_norm.py` — diagnostic: loads a checkpoint, runs 200 RL forward+backward passes (no optimizer step), records (relative_reward, grad_norm) to `grad_norm_log.csv`, plots `grad_norm_analysis.png`; used to confirm round 2 collapse hypothesis
- `--data-suffix` arg (`train_configs.py`, `datasets/apps_dataset.py`, `train.py`) — selects which version of RL data files to load (e.g. `_round1` → `gen_solutions_critic_scores_round1.pkl` + `baseline_solutions_round1.json`)

### `tuning_mode` controls model head
- `'none'` → standard LM head (SFT)
- `'critic'` → 4-class classifier (compile/runtime/failed/passed)
- `'rl'` → policy gradient setup

---

## Known Bugs (fixed)

| Bug | Symptom | Fix |
|---|---|---|
| `((index++))` with `set -e` | Script exits silently when index=0 (arithmetic returns 0=false) | Use `index=$(( index + 1 ))` |
| `generate.py` with `--temperature 0` | `ValueError: temperature must be strictly positive` | Added branch: temp=0 → `do_sample=False` (greedy) |
| `apps_dataset.py` missing baseline file | `FileNotFoundError` on problems without `input_output.json` | Added `os.path.isfile()` guard before `json.load()` |
| `--save_dir exps/my_run` double-prefix | Saves to `exps/exps/my_run/` | `train_configs.py` prepends `exps/` — pass bare name only |
| `test_one_solution.py` infinite loop | One runaway test case stalls the entire 16-worker batch indefinitely (`ulimit -v` caps memory but not CPU) | Added `timeout 300` before all `python test_one_solution.py` calls in every unit test script |
| `--train_path` vs `--train-path` in scripts | `train.py` argparse defines `--train-path` (hyphen); scripts using `--train_path` (underscore) exit silently with error code 0 — pipeline logs "Phase done" and moves on without training | All `train.py` invocations must use `--train-path` |
| Round 1 critic checkpoint path | `checkpoint-8200` doesn't exist; dir only has `checkpoint-8000` | Use `--model_path exps/codet5-small_critic_bs16x16_lr5e-05/checkpoint-8000` |
| Val set bias in `get_dataset` | `val_fnames = fnames[-N:]` on a sorted list always takes the last N problem dirs (4500–4999), all of which have full 20 gen_solutions; first half (0–2499) averages only 18.4 — val is not representative | `train.py`: shuffle `fnames` with `random.Random(42)` before splitting |
| Critic val `prediction_step` discarded preds | `prediction_step` returned `(loss, None, None)` for critic mode — eval loop had no predictions to pass to `compute_metrics`, so accuracy was never computed | Return `(loss, error_preds, error_types)` from critic branch; auto-install `compute_metrics` in `Trainer_RL.__init__` when `tuning_mode='critic'` |

---

## Evaluation Results Summary

> **IMPORTANT:** APPS test split layout — `data/APPS/test/` indices 0–2999 = interview, 3000–3999 = competition, 4000–4999 = introductory. Eval scripts using `-s 0 -e 500` run **interview** problems, NOT introductory. Use `-s 4000 -e 4500` for introductory.

All evals below: interview split (indices 0–499), 500 problems, n=10 samples, temp=0.6.

| Model | pass@1 | pass@5 | pass@10 | CE% | Uniqueness |
|---|---|---|---|---|---|
| SFT baseline | 0.22% | 0.46% | 0.60% | 17.5% | 99.7% |
| **RL round 1 ckpt-3000 (best)** | 0.16% | 0.67% | **1.00%** | 4.1% | 99.0% |
| RL round 1 ckpt-14000 (final) | 0.18% | 0.39% | 0.40% | 78.8% | 34.9% |
| RL round 2 ckpt-1000–5000 (collapsed) | — | — | — | 96–100% | 11–28% |
| Round 5A (old, round2 data) ckpt-2000 | 0.48% | 0.87% | 1.00% | 3.2% | 99.3% |
| Round 5B (round1 data, no filter) | collapsed (CE% 46→67%) | — | — | — | — |
| Round 5C (old, round2 data) | collapsed CE%=100% | — | — | 100% | 10% |
| Round 5A/5C/5E (round1 data) | pending | — | — | — | — |

Full round 1 results (all 14 checkpoints): see `1st_rl_result.md`.  
Round 2 spec and rationale: see `2nd_rl_spec.md`.  
Rounds 2 and 3 session log: see `round2_rl.md`.

> Note: Watcher eval (50 intro problems, n=10) differs from full eval (500 problems, n=10) — pass@k values not directly comparable.

---

## Important Notes

- Use `--db` flag for debug mode (small data splits for quick iteration)
- WandB is integrated in `train.py` for experiment tracking
- DDP (multi-GPU) was attempted and failed due to NCCL ALLGATHER timeout at rank sync — root cause unresolved. Use single-GPU training.
- `torch.multiprocessing.set_sharing_strategy('file_system')` required in training scripts for data loader compatibility
- Unit tests run with `ulimit -v 16000000` (16 GB virtual memory cap) **and** `timeout 300` (5 min per problem) to prevent runaway processes. `ulimit -v` alone does not stop CPU-bound infinite loops.
- `--example_tests 0` uses hidden test cases; `--example_tests 1` uses visible example tests
- The 16-thread unit test loop uses `index=$(( index + 1 ))` not `((index++))` — the latter breaks with `set -e`
- comeback and updated this file **Claude.md** if needed and update `round2_rl.md` when there's progress.
- **CUDA fix (2026-06-17):** `/usr/local/cuda-11.8/.../libcuda.so` (stub) is loaded by default instead of the real driver. Always prepend `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH` to any `python generate.py` or `python train.py` call. Add to `~/.zshrc` to make permanent.