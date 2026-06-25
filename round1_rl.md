# CodeRL Reproduction — Pipeline Notes
> CodeT5-large (770M) · APPS Introductory · REINFORCE + relative returns
> Last updated: 2026-05-05

---

## Quick Reference: Paper Targets vs Our Results

| Model | pass@1 | pass@5 | Eval method |
|---|---|---|---|
| Paper SFT baseline | 6.60% | 8.80% | beam search |
| Paper CodeRL (Lce+Lrl) | 6.20% | **9.39%** | beam search |
| **Our SFT baseline** | **0.22%** | **0.46%** | nucleus sampling, temp=0.6 |
| **Our best RL (ckpt-3000)** | **0.16%** | **0.67%** | nucleus sampling, temp=0.6 |

> Gap vs paper is expected — nucleus sampling ≠ beam search. What matters: **RL must beat SFT**.
> ckpt-3000 achieves pass@10=1.00% vs SFT 0.60% → **+67% relative gain** ✅

---

## Round 1 Results: Full Checkpoint Trajectory

**Setup:** LR=2e-5, batch_size=2, grad_acc=32 (effective=64), 14,990 steps, 10 epochs, single GPU0

| Checkpoint | pass@10 | Uniqueness | CompileErr | Notes |
|---|---|---|---|---|
| SFT baseline | 0.60% | 99.7% | 17.5% | reference |
| ckpt-1000 | 0.80% | 99.7% | 3.7% | healthy ✅ |
| ckpt-2000 | 0.60% | 99.1% | — | healthy ✅ |
| **ckpt-3000** | **1.00%** | **99.0%** | **<5%** | **⭐ peak — use as R2 start** |
| ckpt-4000 | 0.20% | 91.7% | 61.2% | collapse onset ⚠️ |
| ckpt-5000 | 0.00% | 93.6% | 41.3% | deepest collapse ❌ |
| ckpt-6000 | 0.60% | 88.6% | 28.9% | partial recovery |
| ckpt-7000 | 0.40% | 81.5% | 48.3% | re-collapse |
| ckpt-8000 | 0.60% | 77.2% | 30.2% | partial recovery |
| ckpt-9000 | 0.80% | 72.9% | 52.9% | best late recovery |
| ckpt-10000 | 0.40% | 51.2% | 68.2% | severe mode collapse |
| ckpt-11000 | 0.40% | 68.6% | — | — |

**Pattern:** pass@10 and CompileError oscillate anti-correlatively throughout training.
**Uniqueness declines monotonically** (99% → 51%) — model output diversity never recovers after collapse.

---

## Key Findings & Knowhow

### 1. Collapse Root Cause: Fixed Baseline Drift

The SFT greedy baseline never updates as the model improves during RL training.
As the actor improves, relative returns `curr_reward - baseline_reward` become
increasingly negative → training signal is dominated by penalty, not reward.

**Symptoms of collapse (in order of appearance):**
1. CompileError rate spikes past 40%
2. Uniqueness drops (model generates repetitive broken code)
3. pass@k falls to 0
4. `rl_loss` magnitude grows significantly (was -0.08 at step 1500, -4.16 at step 11000)

**Health signals to monitor at each checkpoint:**
- ✅ Healthy: CompileErr <10%, Uniqueness >95%, pass@10 rising
- ⚠️ Warning: CompileErr 20–40%, Uniqueness 85–95%
- ❌ Collapse: CompileErr >50%, Uniqueness <80%, pass@10 = 0

### 2. `--relative_returns` is Non-Negotiable

Without this flag: training collapsed at step ~3000–7000 in all prior runs (99%+ CompileError).
With this flag: model reached ckpt-3000 cleanly before collapsing.

The flag computes `advantage = sampled_reward - baseline_reward` (greedy decode from same prompt).
This centers the reward signal so imperfect-but-improving samples are not over-penalized.

### 3. Data Quality: Round2 Positive Sample Rate

| Dataset | Positive samples (reward > baseline) | Status |
|---|---|---|
| Round 1 (SFT rollouts) | 17.8% | usable |
| Round 2 (RL ckpt-3000 rollouts) | TBD — expect higher | goal |

More positive samples = stronger RL signal = less collapse pressure.

### 4. LR=2e-5 is Too Aggressive

- Collapse onset at ckpt-4000 (~step 3000–4000)
- Round 2 uses LR=1e-5 (halved) from ckpt-3000 starting point
- Fallback values to try if R2 still collapses: LR=5e-6

### 5. Batch Size Constraint (codet5-large on RTX A5000 24GB)

- `batch_size=4` → OOM (22.3 GB, no headroom for FP32 softmax cast)
- `batch_size=2, grad_acc=32` → 23.0 GB, stable
- Always set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## Bugs Fixed (permanent reference)

| Bug | File | Symptom | Fix |
|---|---|---|---|
| `temperature=0` crash | `generate.py` | `ValueError: temperature must be positive` | if temp==0 → `do_sample=False` |
| Missing `--relative_returns` | `train_rl_actor.sh` | collapse at step ~3k–7k | added flag |
| Missing `--model codet5-large` | `train_rl_actor.sh` | trained codet5-base instead | added flag |
| Missing baseline file crash | `datasets/apps_dataset.py:163` | `FileNotFoundError` on problems without `input_output.json` | add `os.path.isfile()` guard |
| `set -e` + `((index++))` silent exit | multiple shell scripts | script dies after first log line, Steps 2/3 never run | `index=$(( index + 1 ))` |
| `--save_dir` double-prefix | `train_rl_actor.sh` | checkpoints saved to `exps/exps/...` | remove leading `exps/` (train_configs.py prepends it) |
| `--train_path` vs `--train-path` | `round2_train_critic.sh`, `round2_train_rl.sh` | argparse error, immediate exit | use hyphen not underscore |
| Stale critic scores after Phase D | pipeline | Phase G used round1 critic scores for round2 solutions | rerun Phase E→F→G chain |
| Infinite-loop test case | unit test scripts | single problem hangs entire 16-worker pool forever | add `timeout 300` to `python test_one_solution.py` |
| Bash inner loop variable collision | `eval_all_ckpts_resume2.sh` | outer `$i` clobbered by inner loop | rename inner vars to `prob`/`idx` |
| DDP NCCL ALLGATHER timeout | `configs/train_configs.py`, `trainer_rl.py` | rank 0 takes 10+ min to reach `_wrap_model()` | unresolved; use single-GPU training |

---

## Pipeline Architecture (Round 2)

```
Phase A  Stochastic sample from rl_actor (ckpt-3000)
         → outputs/sampled_code/codes_round2/

Phase B  Unit tests (16 parallel workers, timeout 300s)
         → outputs/unit_test_score/test_results_round2/

Phase C  Convert to gen_solutions.json
         → data/APPS/train/*/gen_solutions.json

Phase D  Greedy baseline decode (temp=0) from same rl_actor checkpoint
         → outputs/sampled_code/codes_baseline_round2/
         → data/APPS/train/*/baseline_solutions.json

Phase E  Fine-tune critic on round2 data
         → exps/codet5-small_critic_round2/

Phase F  Generate critic scores
         → data/APPS/train/*/gen_solutions_critic_scores.pkl

Phase G  RL training (LR=1e-5, from ckpt-3000)
         → exps/codet5-large_rl_round2/
```

**Critical dependency:** Phase F scores must be regenerated after Phase E.
Using old scores from a different policy = completely wrong rewards.

---

## Checkpoint Inventory

| Path | Description |
|---|---|
| `models/sft_actor/checkpoint-7000` | Correct SFT actor (~10 epochs on APPS train) |
| `models/rl_actor_best_ckpt3000/` | Round 1 best RL checkpoint (pass@10=1.00%) |
| `exps/codet5-large_rl_bs4x16_lr2e-05_newcritic_relreturns/` | Round 1 full RL run (ckpts 5000–14000 survive) |
| `exps/codet5-small_critic_bs32x8_lr5e-05/checkpoint-8000` | Round 1 critic |

> ⚠️ `models/rl_actor/checkpoint-1000` is **mislabeled** — it is actually a step-1000 SFT checkpoint, not an RL checkpoint.

---

## Monitoring Commands

```bash
# RL training health (loss + reward)
grep -E '"loss"|"rl_loss"' logs/train_rl_*.log | tail -20

# Checkpoint appearance
ls exps/codet5-large_rl_round2/ 2>/dev/null

# Eval results summary
cat outputs/unit_test_score/test_results_relrl_ckpt_*/eval_summary.txt 2>/dev/null | grep -A6 "Pass@k"

# GPU status
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader

# Pipeline master watcher
tail -20 logs/pipeline.log
```

---

## Round 2 Status (as of 2026-05-05)

- Phase A–D: ✅ complete
- Phase E (critic fine-tune): 🔄 running on GPU0
- Phase F (critic scores): ⏳ ~03:00 May 6
- Phase G (RL LR=1e-5): ⏳ ~05:00 May 6 start, ~07:00 May 8 finish