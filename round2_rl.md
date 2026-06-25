# CodeRL Round 2 RL — Run Log

**Started:** 2026-05-04 15:56  
**Objective:** Rebuild training data from best RL policy (ckpt-3000), retrain critic on new distribution, restart RL with lower LR to avoid actor–critic mismatch that caused round 1 collapse.

For design rationale, see `2nd_rl_spec.md`. For round 1 results, see `1st_rl_result.md`.

---

## Pipeline Phase Status

| Phase | Description | Status | Time |
|---|---|---|---|
| A | Stochastic sampling from ckpt-3000 (n=20, both GPUs) | ✅ Done | May 4–5, ~6h |
| B | Unit tests on round2 training codes | ✅ Done | May 5, ~4h (incident) |
| C | Convert codes+results → gen_solutions.json | ✅ Done | May 5, ~2s |
| D | Greedy baseline from ckpt-3000 (decode + unit tests + write) | ✅ Done | May 5, ~7h |
| E | Fine-tune critic on round2 data | ✅ Done | Run 2 complete May 6; 5 epochs; best val_loss=0.457 @ epoch 1 |
| F | Generate critic scores with new critic | ✅ Done | Complete 14:00 May 6; 4805 pkl files; checkpoint-500 |
| G | RL training from ckpt-3000, LR=1e-5 | ❌ Killed | Collapsed by step 1000 (CE%=100%, dashes); killed May 7 at step ~5000 |

**Phase E run 2 PID:** 645581 (`logs/train_critic_round2.log`) — complete  
**Phase F PID:** 655316 (`logs/round2_critic_scores.log`) — complete  
**Phase G PID:** 656321 (`logs/train_rl_round2.log`) — killed May 7 (collapsed)  
**Watcher PID:** 669378 (`logs/watch_rl_eval.log`) — killed May 7

---

## Phase Outputs

| Output | Count | Notes |
|---|---|---|
| `outputs/sampled_code/codes_round2/` | 5,000 files | ckpt-3000, temp=0.6, n=20 |
| `outputs/unit_test_score/test_results_round2/` | 4,804 files | 196 skipped (no input_output.json) |
| `data/APPS/train/*/gen_solutions.json` | 4,803 written | 197 skipped |
| `outputs/sampled_code/codes_baseline_round2/` | 5,000 files | ckpt-3000, temp=0, n=1 |
| `outputs/unit_test_score/test_results_baseline_round2/` | 4,805 files | |
| `data/APPS/train/*/baseline_solutions.json` | overwritten (round1 backed up) | |
| `exps/codet5-small_critic_round2_bs32x8_lr2e-05/` | checkpoint-500…3500 (7 ckpts) | Phase E run 2 complete; best val_loss at epoch 1 → use checkpoint-500 for Phase F |
| `data/APPS/train/*/gen_solutions_critic_scores.pkl` | pending regen | Phase F |
| `exps/codet5-large_rl_round2_bs4x16_lr1e-05_relreturns/` | not started | Phase G |

---

## Bugs Found and Fixed During This Run

### 1. Infinite-loop test case stalls unit test batch (Phase B)
- **Symptom:** PID 130146 (`test_one_solution.py --i 3345`) ran for 9+ hours with CPU at 100%. The 16-worker batch `wait` barrier blocked indefinitely. `ulimit -v 16000000` caps memory but not CPU time.
- **Fix:** Killed stuck process. Added `timeout 300` before `python test_one_solution.py` in all 12 unit test scripts.
- **Impact:** Problem 3345 has no test result (acceptable, need ≥4,700).

### 2. `--train_path` vs `--train-path` (Phases E and G)
- **Symptom:** `train.py` argparse defines `--train-path` (hyphen). Scripts had `--train_path` (underscore) → argparse error, exit code 0, pipeline logged "Phase done" and moved on silently without training.
- **Fix:** Changed `--train_path` → `--train-path` in `round2_train_critic.sh` and `round2_train_rl.sh`.
- **Rule:** All `train.py` invocations must use `--train-path` (hyphen).

### 3. Wrong critic checkpoint path (Phase E)
- **Symptom:** Script pointed to `checkpoint-8200` but dir only contains `checkpoint-7000` and `checkpoint-8000`.
- **Fix:** Changed `--model_path` to `exps/codet5-small_critic_bs16x16_lr5e-05/checkpoint-8000`.

### 4. Stale critic scores data integrity risk (Phase F→G)
- **Symptom:** Phase F ran twice with empty model_path (due to Bug #2+#3) → crashed with TypeError → no new pkl written. The `gen_solutions_critic_scores.pkl` files still held **round 1 scores indexed to round 1 solutions**, but `gen_solutions.json` had been overwritten with round 2 data. Using these in RL training would produce completely wrong reward signals.
- **Fix:** Stopped Phase G after 1 step (no checkpoints saved). Deleted incomplete RL dir. Relaunched E→F→G only after Phase E was confirmed working.

### 6. `generate.py` crashes with `output_path=None` in critic-scores mode (Phase F)
- **Symptom:** `generate.py` line 96 calls `os.path.exists(args.output_path)` unconditionally. When `--critic_scores` is set, `--output_path` is not provided (critic scores write directly to per-problem pkl, not to an output dir) → `TypeError: stat: path should be string… not NoneType`. Phase F exited in <1s. The EFG chain's `bash script | tee` pipeline masked the non-zero exit code, so the chain silently moved on to Phase G.
- **Fix:** Wrapped the output_path block in `generate.py` with `if not args.critic_scores:` guard. Phase F script already updated to use `checkpoint-2000`.
- **Impact:** Phase G started with stale round-1 critic scores (same risk as Bug #4). Killed at step 912 (no checkpoints saved). Incomplete RL dir deleted. F→G awaiting relaunch.

### 5. `Trainer_RL.prediction_step` incompatible with eval (Phase E validation)
- **Symptom:** `compute_loss` has a non-standard `step` positional argument; `prediction_step` called it as `compute_loss(model, inputs, return_outputs=True)` — missing `step` → would crash with `TypeError` on first eval pass.
- **Fix:** Added critic-specific early-return at top of `prediction_step` in `trainer_rl.py`: calls model directly, returns `(loss.mean().detach(), None, None)`. Also added `--val-split` arg to `train_configs.py`, split logic to `train.py`, and `evaluation_strategy='epoch'` when val data present.
- **Impact:** Phase E relaunched with `--val-split 500` (dirs 4500–4999 held out). 3,775 total steps vs previous 4,165.

### 7. Val set bias — unshuffled split always takes last N dirs
- **Symptom:** `get_dataset` does `fnames = sorted(os.listdir(...))` then `val_fnames = fnames[-500:]`. Sorted order means val is always dirs 4500–4999. Problems 2500–4999 all have exactly 20 gen_solutions; problems 0–2499 average only 18.4 (some missing). Val set was entirely from the "full" half — not representative.
- **Fix:** Added `random.Random(42).shuffle(fnames)` in `train.py` before the val split (only when `val_split > 0 and not args.db`).
- **Impact:** Phase E restarted from scratch with shuffled val set.

### 9. Round 2 RL collapse at step 1000 — starting from rl_actor_best_ckpt3000 too fragile
- **Symptom:** Watcher eval at checkpoint-1000 and checkpoint-2000 both show CE%=100%, pass@10=0%, uniqueness=11%. Generated code is entirely `----...` (repeated dashes). LM loss appeared healthy (~0.19) throughout, masking the collapse.
- **Root cause (hypothesis):** rl_actor_best_ckpt3000 was already at round 1's stability boundary (round 1 peaked at step 3000 then immediately began collapsing). A second round of RL — even at LR=1e-5 — pushed it over the edge immediately. The alternating LM/RL scheme kept LM cross-entropy low on memorised training codes while RL updates destroyed generalisation to new prompts. 81% of relative-return samples were negative (77,944 negative vs 2,004 positive), creating a constant pushdown signal with almost no positive signal to steer toward.
- **Contrast with round 1:** Round 1 started from the SFT actor (fresh weights) and took ~7000 steps to collapse. Round 2 started from an already-RL-trained model and collapsed in <1000 steps.
- **Impact:** Phase G training is useless from step 1000 onward. No preserved checkpoints.
- **Fix options:**
  1. Restart RL from SFT actor (`models/sft_actor/`) instead of rl_actor_best_ckpt3000
  2. Use much lower LR (5e-6 or lower)
  3. Add KL penalty to prevent distribution shift from the reference model
  4. Reduce the negative-reward dominance: filter training data to problems where at least some solutions pass, or clip rewards

### 8. Critic val accuracy never computed — `prediction_step` discarded predictions
- **Symptom:** Critic `prediction_step` returned `(loss, None, None)`. With `compute_metrics=None`, the eval loop set `prediction_loss_only=True`, so no predictions were collected. Only `eval_loss` was logged; `eval_acc` was never computed.
- **Fix:** Critic `prediction_step` now returns `(loss, error_preds, error_types)` (respects `prediction_loss_only`). `Trainer_RL.__init__` auto-installs `compute_metrics = lambda ep: {"eval_acc": (ep.predictions == ep.label_ids).mean()}` when `tuning_mode='critic'` and no custom function is provided.
- **Impact:** Phase E restarted; `eval_acc` now logged alongside `eval_loss` after each validation epoch.

---

## Phase E Config (Critic Fine-tuning)

### Run 1 (completed May 6 03:24, stale — val set biased, no eval_acc)

| Parameter | Value |
|---|---|
| Init from | `exps/codet5-small_critic_bs16x16_lr5e-05/checkpoint-8000` |
| Save dir | `exps/codet5-small_critic_round2_bs32x8_lr2e-05/` |
| Val split | last 500 dirs (4500–4999) — **biased, not representative** |
| Steps | 3,775 total |
| Save freq | 1000; save_total_limit=2 (ckpt-1000 pruned) |

#### Run 1 — Eval Loss by Epoch

| Epoch | Train loss (≈) | Val loss | Checkpoint saved |
|---|---|---|---|
| 1 | ~0.48 | 0.4841 | checkpoint-1000 (**pruned**) |
| 2 | ~0.41 | 0.4866 | checkpoint-2000 |
| 3 | ~0.41 | 0.4954 | checkpoint-3000 |
| 4 | ~0.40 | 0.5112 | (no checkpoint) |
| 5 | ~0.43 | 0.5611 | final_checkpoint |

Overfitting confirmed: val loss rose monotonically. Best epoch was 1 (pruned). Discarded due to bugs #7 and #8.

### Run 2 (started May 6, in progress — shuffled val, eval_acc tracked)

| Parameter | Value |
|---|---|
| Init from | `exps/codet5-small_critic_bs16x16_lr5e-05/checkpoint-8000` |
| Save dir | `exps/codet5-small_critic_round2_bs32x8_lr2e-05/` |
| LR | 2e-5 |
| Epochs | 5 |
| Batch | 32 per replica × 8 grad-acc = 256 effective |
| Val split | random 500 problems (seed=42 shuffle); eval_loss + eval_acc logged each epoch |
| Save freq | 500 steps; save_total_limit=20 |
| GPU | GPU0 |

#### Run 2 — Eval by Epoch

| Epoch | Train loss (≈) | Train acc | Val loss | Val acc | Nearest checkpoint |
|---|---|---|---|---|---|
| 1 | ~0.45 | ~82% | **0.4569** | **81.8%** | checkpoint-500 ← use for Phase F |
| 2 | ~0.43 | ~83% | 0.4633 | 81.7% | checkpoint-1000 |
| 3 | ~0.41 | ~84% | 0.4764 | 81.4% | checkpoint-1500/2000 |
| 4 | ~0.40 | ~84% | 0.4690 | 82.1% | checkpoint-2500/3000 |
| 5 | ~0.39 | ~84% | 0.4939 | 81.6% | checkpoint-3500 |

Train acc 0.844 (final), train loss 0.391. Overfitting is mild vs run 1 (val_loss +8% run 2 vs +16% run 1 over 5 epochs). Best checkpoint is **checkpoint-500** (closest to epoch 1 eval; epoch 1 model is evaluated just before step 748, with no exact save — checkpoint-500 is the last save before that eval).

## Phase G Config (RL Training)

| Parameter | Value |
|---|---|
| Warmstart | `models/rl_actor_best_ckpt3000` |
| Save dir | `exps/codet5-large_rl_round2_bs4x16_lr1e-05_relreturns/` |
| LR | **1e-5** (halved from round 1's 2e-5) |
| Epochs | 10 (~14,990 steps) |
| Batch | 2 per replica × 32 grad-acc = 64 effective |
| `--relative_returns` | yes |
| `save_total_limit` | 15 |
| GPU | GPU0 |

---

## Monitoring

```bash
# Phase E critic training progress
tail -f logs/train_critic_round2.log | grep -E "loss|step|epoch"

# Phase E→F→G chain log
tail -f logs/round2_efg.log

# Phase G RL training
tr '\r' '\n' < logs/train_rl_round2.log | grep "^{'loss'" | tail -5

# Watcher progress (live eval on GPU1)
tail -f logs/watch_rl_eval.log

# Watcher eval results per checkpoint
cat outputs/unit_test_score/test_results_round2_watch_ckpt_<N>/eval_summary.json

# Preserved checkpoints (beat SFT baseline)
ls models/rl_round2_best/
```

#### Phase G — Live Watcher Results (50 intro test problems, n=10)

| Checkpoint | pass@10 | CE% | Uniqueness | Status |
|---|---|---|---|---|
| ckpt-1000 | 0.00% | 100.0% | 11.4% | ❌ Collapsed (all dashes) |
| ckpt-2000 | 0.00% | 100.0% | 11.4% | ❌ Collapsed |
| ckpt-3000 | 0.00% | 96.6% | 13.6% | ❌ Collapsed |
| ckpt-4000 | 0.00% | 100.0% | 10.0% | ❌ Collapsed (worsening) |
| ckpt-5000 | 0.00% | 99.8% | 11.4% | ❌ Collapsed |
| **round1 ckpt-3000** | **0.00%** | **2.4%** | **98.8%** | ✅ Healthy (0/50 solved; within variance for 1% pass@10) |

**Conclusion:** Round 2 model collapsed from step 1000. Round 1 ckpt-3000 on same 50 problems: CE=2.4%, uniqueness=98.8% — diverse real Python code, just hard problems (0.5 solves expected in 50 problems at 1% pass@10). Root cause of round 2 collapse: starting from rl_actor_best_ckpt3000 (already at round 1's stability cliff) — see Bug #9.

**Phase G health signals (from watcher eval_summary.json):**
- ✅ Healthy: CE% <20%, uniqueness >80%
- ⚠️ Warning: CE% 20–50%, uniqueness declining
- ❌ Collapse: CE% >80%, uniqueness <50%

**Preservation thresholds:** pass@10 > 0.60% AND CE% < 17.5% → copied to `models/rl_round2_best/`

---

## Timeline

| Time | Event |
|---|---|
| May 4 15:56 | Phase A started (both GPUs) |
| May 4 ~17:00 | GPU1 done (probs 2500–4999) |
| May 5 ~00:00 | GPU0 done (probs 0–2499), Phase A complete |
| May 5 ~03:32 | Phase B stuck on problem 3345 (infinite loop) |
| May 5 13:35 | Killed stuck worker; Phase B resumed |
| May 5 14:16 | Phase B complete (4,804 results); Phase C done in ~2s |
| May 5 14:16 | Phase D started (greedy baseline, both GPUs) |
| May 5 ~18:10 | Phase D step 1 (decode) complete |
| May 5 ~20:30 | Phase D complete (baseline_solutions.json written) |
| May 5 17:10 | Phase E/F/G all failed silently (bugs #2+#3+#4) |
| May 5 21:19 | Bugs identified and fixed |
| May 5 21:35 | Phase E correctly relaunched (no val split) |
| May 5 21:52 | Phase E killed and relaunched again with --val-split 500 (bugs #5) |
| May 6 ~01:00 | Phase E epoch 1 done; val_loss=0.4841 |
| May 6 ~03:00 | Phase E epoch 2 done; val_loss=0.4866 |
| May 6 ~05:00 | Phase E epoch 3 done; val_loss=0.4954 |
| May 6 ~07:00 | Phase E epoch 4 done; val_loss=0.5112; step 3157/3775 |
| May 6 03:24 | Phase E run 1 complete (5 epochs; val_loss: 0.484→0.487→0.495→0.511→0.561) |
| May 6 03:25 | Phase F crashed (Bug #6 output_path=None); Phase G started on stale scores |
| May 6 ~09:00 | Phase G killed at step 912 (no ckpts); generate.py fixed; F→G awaiting relaunch |
| May 6 ~07:30 | Phase F expected complete (run 1 critic) |
| May 6 ~07:30 | Phase G RL training starts (run 1 critic) |
| May 6 | Bugs #7 (val bias) and #8 (no eval_acc) found and fixed |
| May 6 | Phase E run 2 started: shuffled val split, eval_acc tracking, save every 500 steps |
| May 6 | Phase E run 2 complete (5 epochs, ~5.5h); best val_loss=0.457 @ epoch 1; checkpoint-500 selected for Phase F |
| May 6 14:00 | Phase F complete; 4805 pkl files; checkpoint-500 |
| May 6 14:00 | Phase G started (LR=1e-5, rl_actor_best_ckpt3000) |
| May 6 ~20:00 | Phase G epoch 1 complete (~step 1500); LM loss stable ~0.19; RL loss ~-1.2 to -1.5; checkpoint-1000 saved |
| May 6 21:38 | Watcher launched (PID 669378); evaluating ckpt-1000 on GPU1 now |
| May 6 21:50 | Watcher: ckpt-1000 pass@10=0%, CE=100% — collapsed |
| May 6 22:02 | Watcher: ckpt-2000 pass@10=0%, CE=100%, uniqueness=11.4% — all dashes, fully collapsed |
| May 7 ~07:00 | Watcher: ckpt-3000 pass@10=0%, CE=96.6%, uniqueness=13.6% — still collapsed |
| May 7 ~07:20 | Watcher: ckpt-4000 pass@10=0%, CE=100%, uniqueness=10.0% — collapsed, worsening |
| May 7 07:53 | Watcher: ckpt-5000 pass@10=0%, CE=99.8% — collapsed |
| May 7 07:53 | round1 ckpt-3000 eval on same 50 problems: CE=2.4%, uniqueness=98.8%, pass@10=0% (within variance) |
| May 7 | Phase G killed; both GPUs free; round 3 planned: RL from SFT using existing round 2 data |

---

## Eval Plan (after Phase G)

Round 2 checkpoints all collapsed (CE%=96–100%) — no eval performed beyond watcher results above.

---

# Round 3 — RL from SFT with Round 1 Data

**Started:** 2026-05-25  
**Objective:** Fix root cause of round 2 collapse. Diagnostic work confirmed the collapse was caused by data quality, not just the warm-start model.

## Diagnosis (2026-05-25)

### Grad norm analysis (`diagnose_grad_norm.py`)
Ran 200 RL forward+backward passes from `rl_actor_best_ckpt3000` on round 2 data (no optimizer step). Results:

| Reward group | n | Median grad norm | Mean | Max |
|---|---|---|---|---|
| Positive | 11 | 0.51 | 2.46 | 12.46 |
| Zero | 5 | 0.00 | 0.00 | 0.00 |
| **Negative** | **184** | **1.48** | **8.41** | **176.80** |

Negative-reward batches produce **3–14× larger gradients** than positive. Combined with 81% negative-reward samples in round 2 data, the RL update signal is a nearly constant high-magnitude pushdown with almost no positive guidance.

### Round 1 vs Round 2 data comparison

| | Round 1 (SFT samples) | Round 2 (ckpt-3000 samples) |
|---|---|---|
| Pass rate (stochastic, temp=0.6) | **32.2%** (30,886/95,983) | **0%** |
| Positive relative returns | **13.2%** | 2.1% |
| Negative relative returns | 17.8% | **81.1%** |
| Max positive reward | +2.0 | +0.4 only |

Root cause of entropy collapse: ckpt-3000's output distribution became too narrow (peaked). Greedy decode (temp=0) still hits the peak → good eval results. Stochastic sampling (temp=0.6) escapes the peak → 0% pass rate. Round 2 RL training had no positive signal to learn from.

### Why ckpt-3000 eval > SFT despite worse stochastic samples
ckpt-3000 has lower pass@1 (0.16% < SFT 0.22%) but higher pass@10 (1.00% > SFT 0.60%). The RL training sharpened the distribution so greedy/best-of-N performance improved, but individual sample diversity collapsed. SFT's broader distribution produces consistently adequate samples; ckpt-3000's narrow distribution either nails it or fails completely.

## Round 3 Config

| Parameter | Value | Rationale |
|---|---|---|
| Warmstart | `models/sft_actor/` | Healthy entropy, broad distribution |
| Data | round 1 (`--data-suffix _round1`) | 13.2% positive, 32% pass, full reward range |
| Critic scores | `gen_solutions_critic_scores_round1.pkl` | Round 1 critic (checkpoint-8000) |
| Baseline | `baseline_solutions_round1.json` | SFT greedy baseline |
| LR | 1e-5 | Half of round 1; same as round 2 attempt |
| Save dir | `exps/codet5-large_rl_round3_sft_lr1e-05_relreturns/` | |
| Epochs | 10 (~14,990 steps) | |
| Batch | 2 per replica × 32 grad-acc = 64 effective | |
| save_total_limit | 15 | Keeps all 15 checkpoints (14,990/1000=~15) |
| GPU | GPU0 | |

**Training PID:** 3884625 (`logs/train_rl_round3.log`)  
**Watcher PID:** 3937358 (`logs/watch_rl_eval_round3.log`) — GPU1, 50 intro problems, n=10

## Code Changes for Round 3

- Added `--data-suffix` arg to `train_configs.py`, `datasets/apps_dataset.py` (`__init__` + f-string filenames), `train.py` (`make_dataset`)
- New scripts: `scripts/round3_train_rl.sh`, `scripts/watch_rl_eval_round3.sh`
- Cleaned up `outputs/archive/` (moved pre-round1 experiment dirs)
- Deleted `exps/exps/` (double-prefix bug artifact), `models/rl_round2_best/` (no valid ckpts)

## Monitoring

```bash
# Training loss
tr '\r' '\n' < logs/train_rl_round3.log | grep "^{'loss'" | tail -5

# Watcher eval results
tail -f logs/watch_rl_eval_round3.log

# Per-checkpoint eval
cat outputs/unit_test_score/test_results_round3_watch_ckpt_<N>/eval_summary.json
```

## Early Training Signals (step ~600)

- RL loss: −0.02 to −0.06 (vs round 2's −1.2 to −1.5 at same stage — 20× smaller, healthy)
- LM loss: 0.19 → 0.22 (slight upward drift, expected during RL updates)

## Timeline

| Time | Event |
|---|---|
| May 25 | Grad norm diagnosis run; confirmed negative-reward gradient dominance in round 2 data |
| May 25 | Round 1 vs round 2 data distribution comparison; confirmed 0% pass rate in round 2 samples |
| May 25 | `--data-suffix` added; `round3_train_rl.sh` + `watch_rl_eval_round3.sh` created |
| May 25 ~19:47 | Round 3 training started (PID 3884625, GPU0) |
| May 25 ~22:44 | Watcher started (PID 3937358, GPU1) |
| May 25 ~step 600 | RL loss −0.02 to −0.06, LM loss 0.20–0.22 — healthy so far |
| May 26 | Round 3 killed at step ~7000; collapse confirmed between ckpt-3000 and ckpt-4000 |

## Watcher Results

| Checkpoint | pass@10 | CE% | FailedTest | RuntimeError | Uniqueness |
|---|---|---|---|---|---|
| 1000 | 0% | 5.6% | 64.2% | 30.2% | 100% |
| 2000 | 0% | 6.8% | 63.4% | 29.8% | 99.8% |
| 3000 | 0% | 6.9% | 64.6% | 28.5% | 100% |
| **4000** | **0%** | **53.8%** | 29.6% | 16.6% | 99.8% |
| 5000 | 0% | 52.0% | 33.2% | 14.8% | 99.8% |
| 6000 | 0% | 44.8% | 31.8% | 23.4% | 100% |
| 7000 | 0% | 45.4% | 36.2% | 18.4% | 99.0% |

**Result: COLLAPSED** — same pattern as Round 1. CE% jumped from 6.9% → 53.8% between ckpt-3000 and ckpt-4000. ckpt-3000 is best.

## Round 3 Collapse Analysis

### Pattern
Identical to Round 1 collapse:
- Steps 1000–3000: stable, CE% ≈ 6–7%, training loss ≈ 0.18–0.20
- Steps 3000–4000: **gradient explosion** — per-layer gradient norms spike to ±1000–10000 in WandB
  - `encoder.final_layer_norm.weight`: ±10000
  - `encoder.block.1.layer.1.DenseReluDense.wi.weight`: ±5000
  - `decoder.block.*.layer_norm.weight`: ±3000–4000
  - Layer norm gradients exploding is especially severe (normally most stable)
- Step 4000+: CE% stabilizes at ~50%, model generates mostly non-compilable code

### Root Cause: max_grad_norm=1.0 insufficient
- `TrainingArguments` default `max_grad_norm=1.0` — never explicitly set in any training script
- Raw gradients reached ±10000; clipping to L2-norm=1.0 scales down the magnitude but the update direction is unchanged
- Gradient explosion happened around step 3000; CE loss lag of ~500 steps before visible in eval
- Reducing LR from 2e-5 (round 1) → 1e-5 (round 3) did NOT delay or prevent collapse — collapse point identical

### What this tells us
The collapse is not LR-driven. It is caused by periodic large-gradient batches (likely correlated with certain training data batches) that overwhelm even grad-clipped updates. The fix must tighten gradient clipping AND reduce step size.

## Round 4 Plan

**Hypothesis:** `max_grad_norm=0.1` (10× tighter clipping) + `lr=5e-6` (2× smaller step) will prevent gradient explosion past step 3000.

| Parameter | Round 1 | Round 3 | **Round 4** |
|---|---|---|---|
| Warmstart | SFT | SFT | SFT |
| Data | round 1 | round 1 | round 1 |
| LR | 2e-5 | 1e-5 | **5e-6** |
| max_grad_norm | 1.0 (default) | 1.0 (default) | **0.1** |
| Collapse step | ~3000 | ~3000 | TBD |

Script: `scripts/round4_train_rl.sh`
Save dir: `exps/codet5-large_rl_round4_sft_lr5e-06_gnorm0.1_relreturns/`

## Success Criteria vs Round 1

- Peak pass@10 > 1.00%
- Collapse boundary beyond step 3000 (ideally none through step 14000)
- Uniqueness stays >80% past step 5000
