# CodeRL Reproduction — Round 2 RL Fine-tuning Specification

> **Outcome (2026-06-23): Round 2 failed. RL training collapsed at step 1000 with CE%=100%.  
> Root cause: ckpt-3000 rollouts had ~0% pass rate → 97.9% negative relative returns → negative gradients dominated → catastrophic forgetting.  
> See `round2_rl.md` for full post-mortem and `round5_rl.md` for the fix (`--positive_only`).**

**Date planned:** 2026-05-04  
**Motivation:** Round 1 peaked at ckpt-3000 then collapsed due to actor–critic distribution mismatch. Round 2 closes that gap by rebuilding all training data from ckpt-3000 rollouts.

---

## Hypothesis

The round 1 critic was trained on SFT actor rollouts. Once the RL policy diverged from SFT (around step 3000), the critic's reward estimates no longer reflected the RL policy's output distribution — producing noisy, unreliable gradients that caused the oscillating collapse.

**Round 2 fix:** Use ckpt-3000 (the best RL policy) to regenerate the training dataset, retrain the critic on this new distribution, and restart RL from ckpt-3000. This aligns actor and critic distributions at the start of round 2 training.

---

## Changes vs Round 1

| Dimension | Round 1 | Round 2 |
|---|---|---|
| Actor warmstart | `models/sft_actor` | `models/rl_actor_best_ckpt3000` |
| Training data source | SFT actor rollouts | ckpt-3000 rollouts |
| Critic training data | SFT actor rollouts | ckpt-3000 rollouts |
| Critic init | codet5-small (scratch) | fine-tune from round1 critic ckpt-8200 |
| Baseline source | SFT greedy decode | ckpt-3000 greedy decode |
| RL learning rate | 2e-5 | **1e-5** (halved — key change) |
| Critic learning rate | 5e-5 | 2e-5 (fine-tuning rate) |
| Critic epochs | 10 | 5 |
| RL save_total_limit | 10 | **15** (keep more to avoid pruning peak ckpt) |

### Why LR=1e-5 for round 2 RL

Round 1 showed a sharp collapse between ckpt-3000 and ckpt-4000 (CE jumped from 4% to 61% in a single 1000-step window). The subsequent oscillation pattern (CE alternating between 30–80% for the rest of training) is characteristic of LR overshooting the reward signal gradient. Halving the LR keeps the policy in the stable region longer and gives the reward signal more time to guide learning before entropy collapse.

### Why fine-tune critic rather than retrain from scratch

The round 1 critic already learned useful code syntax and structure features from 4,802 problems × 20 samples. Fine-tuning from `checkpoint-8200` with a lower LR (2e-5 vs 5e-5) will adapt the reward head to ckpt-3000's output distribution faster than re-learning from scratch, with less risk of overfitting to the smaller round 2 dataset.

### Why save_total_limit=15 instead of 10

Round 1's peak checkpoint (ckpt-3000) was pruned mid-training and had to be manually saved by the user. With save_total_limit=15, the first 15 checkpoints (steps 1000–15000) would all survive if training completes at step ~14990 again. This ensures the peak checkpoint is never lost.

---

## Training Configuration

### Phase E — Round 2 Critic Fine-tuning

| Parameter | Value |
|---|---|
| Architecture | CodeT5-small |
| Init from | `exps/codet5-small_critic_bs16x16_lr5e-05/checkpoint-8200` |
| Tuning mode | `critic` |
| Training data | Round 2 `gen_solutions.json` (ckpt-3000 rollouts) |
| Batch size | 32 per replica × 8 grad-acc = 256 effective |
| Learning rate | 2e-5 |
| Epochs | 5 |
| Save dir | `exps/codet5-small_critic_round2_bs32x8_lr2e-05/` |
| GPU | GPU0 |

### Phase G — Round 2 RL Training

| Parameter | Value |
|---|---|
| Architecture | CodeT5-large (770M params) |
| Warmstart | `models/rl_actor_best_ckpt3000` |
| Tuning mode | `rl` with `--relative_returns` |
| Critic | `exps/codet5-small_critic_round2_bs32x8_lr2e-05/` (best checkpoint) |
| Batch size | 2 per replica × 32 grad-acc = 64 effective |
| **Learning rate** | **1e-5** |
| Epochs | 10 |
| Save dir | `exps/codet5-large_rl_round2_bs4x16_lr1e-05_relreturns/` |
| Save freq | every 1,000 steps |
| save_total_limit | **15** |
| GPU | GPU0 (RTX A5000 24 GB) |
| FP16 | yes |

---

## Dataset for Round 2

| File | Source | Count (expected) |
|---|---|---|
| `outputs/sampled_code/codes_round2/` | ckpt-3000, temp=0.6, n=20 | ~4,800 files |
| `outputs/unit_test_score/test_results_round2/` | hidden unit tests (train set) | ~4,800 pkl |
| `data/APPS/train/*/gen_solutions.json` | converted from above | ~4,800 problems |
| `outputs/sampled_code/codes_baseline_round2/` | ckpt-3000, temp=0, n=1 | ~4,800 files |
| `outputs/unit_test_score/test_results_baseline_round2/` | hidden unit tests | ~4,800 pkl |
| `data/APPS/train/*/baseline_solutions.json` | converted from above | ~4,800 problems |
| `data/APPS/train/*/gen_solutions_critic_scores.pkl` | round2 critic inference | ~4,800 problems |

**Backup policy:** All overwritten files (`gen_solutions.json`, `baseline_solutions.json`, `gen_solutions_critic_scores.pkl`) are backed up in-place with `_round1` suffix before overwriting. The pipeline skips backup if `_round1` already exists (safe to re-run).

---

## Pipeline Scripts

| Phase | Script | Est. time |
|---|---|---|
| A — Sampling | `scripts/round2_sample_actor.sh` | ~4–5 h (both GPUs) |
| B — Unit tests (train) | `scripts/round2_unit_tests_train.sh` | ~2–3 h (CPU) |
| C — Convert → gen_solutions | `run_round2_pipeline.sh` inline | ~30 min |
| D — Baseline | `scripts/round2_baseline.sh` | ~5–6 h |
| E — Critic training | `scripts/round2_train_critic.sh` | ~4–6 h |
| F — Critic scores | `scripts/round2_critic_scores.sh` | ~2 h |
| G — RL training | `scripts/round2_train_rl.sh` | ~50–55 h |
| **Master** | `scripts/run_round2_pipeline.sh` | **~68–78 h total** |

Run everything with:
```bash
bash scripts/run_round2_pipeline.sh > logs/round2_pipeline.log 2>&1 &
tail -f logs/round2_pipeline.log
```

The master script is idempotent — each phase checks for existing output and skips if already done.

---

## Monitoring

```bash
# Overall pipeline progress
tail -f logs/round2_pipeline.log

# Phase A sampling progress (both GPUs)
tail -f logs/round2_sample_gpu0.log
tail -f logs/round2_sample_gpu1.log
watch -n30 'echo "GPU0: $(ls outputs/sampled_code/codes_round2/ | wc -l)/5000"'
# Phase E critic training
tail -f logs/train_critic_round2.log | grep -E "loss|epoch|step"

# Phase G RL training
tail -f logs/train_rl_round2.log | grep -E "loss|reward|epoch|step"

# Checkpoint health check (run periodically during Phase G)
ls exps/codet5-large_rl_round2_bs4x16_lr1e-05_relreturns/
```

**Health signals during Phase G RL training:**
- ✅ Healthy: CE rate <20%, uniqueness >80%, reward slowly rising from ~0
- ⚠️ Early warning: CE rate 20–50%, uniqueness declining
- ❌ Collapse: CE rate >80%, uniqueness <50%, code length at median=510 (template mode)

---

## Expected Outcomes

Given the fixes (aligned actor–critic distribution, halved LR, better warmstart), round 2 should:

1. **Maintain healthy training longer** — the collapse boundary should move past step 3000 (ideally no collapse at all, or collapse much later)
2. **Higher peak pass@10** — starting from ckpt-3000 means the policy already knows how to pass some tests; the new critic should provide a more accurate reward signal for further improvement
3. **Less oscillation** — lower LR reduces gradient noise amplification

If round 2 also collapses, the next debugging step would be reducing LR further (5e-6) or investigating entropy regularization (adding KL penalty to SFT to prevent diversity loss).

---

## Eval Pipeline for Round 2 Checkpoints

After Phase G completes (or during, for intermediate checkpoints), evaluate with a modified version of `eval_relrl_ckpts.sh`:

```bash
# Quick modification: change CKPT_BASE in eval_relrl_ckpts.sh
CKPT_BASE="exps/codet5-large_rl_round2_bs4x16_lr1e-05_relreturns"
```

Or run inline:
```bash
bash scripts/eval_relrl_ckpts.sh  # after updating CKPT_BASE at the top
```

Results will be written to `outputs/unit_test_score/test_results_relrl_round2_ckpt_*/`.
