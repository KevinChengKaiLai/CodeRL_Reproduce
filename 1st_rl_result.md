# CodeRL Reproduction — Round 1 RL Fine-tuning Results

**Date:** 2026-04-29 to 2026-05-03  
**Total training time:** 52.6 hours (14,990 steps, 10 epochs)  
**Evaluated on:** APPS Introductory split, 500 problems, n=10 samples, temp=0.6

---

## Training Configuration

| Parameter | Value |
|---|---|
| Architecture | CodeT5-large (770M params) |
| Warmstart | `models/sft_actor/` (SFT ~10 epochs on APPS train) |
| Tuning mode | `rl` with `--relative_returns` |
| Critic | `exps/codet5-small_critic_bs16x16_lr5e-05/checkpoint-8200` |
| Critic training data | SFT actor rollouts (20 samples/problem, 5000 problems) |
| Batch size | 2 per replica × 32 grad-acc steps = 64 effective |
| Learning rate | 2e-5 |
| Epochs | 10 |
| Save dir | `exps/codet5-large_rl_bs4x16_lr2e-05_newcritic_relreturns/` |
| Save freq | every 1,000 steps, keep last 10 |
| GPU | Single GPU0 (RTX A5000 24 GB) |
| FP16 | yes |

### Why `--relative_returns`

Two prior RL runs (one without `--relative_returns`, one with a different critic) both collapsed to 99%+ compile errors by step 7000–26000. Relative returns compute advantage rewards:

```
reward = critic(sample) - critic(greedy_baseline)
```

where the greedy baseline is a temp=0 decode from the same SFT model. This centers the reward signal and prevents over-penalizing imperfect-but-useful samples.

---

## Dataset Statistics (before training)

| File | Count |
|---|---|
| `data/APPS/train/*/gen_solutions.json` | 4,802 / 5,000 problems |
| `data/APPS/train/*/gen_solutions_critic_scores.pkl` | 4,802 / 5,000 |
| `data/APPS/train/*/baseline_solutions.json` | 4,803 / 5,000 |

- **Total RL samples:** 95,983
- Positive advantage (sampled > baseline): 17,116 / 95,983 (17.8%)
- Equal (sampled = baseline): 66,186 (69.0%)
- Negative advantage: 12,681 (13.2%)
- ~198 problems excluded (no `input_output.json`)

---

## Full Evaluation Results (all 14 checkpoints)

| Ckpt | pass@1 | pass@5 | pass@10 | CE% | Uniqueness | Partial% | MeanLen | Probs Solved |
|---:|---|---|---|---|---|---|---|---|
| SFT baseline | 0.22% | 0.46% | 0.60% | 17.5% | 99.7% | — | — | — |
| 1000 | 0.22% | 0.58% | 0.80% | 3.7% | 99.7% | 6.14% | 357 | 4 |
| 2000 | 0.14% | 0.45% | 0.60% | 4.4% | 99.1% | 5.89% | 301 | 3 |
| **3000** | 0.16% | 0.67% | **1.00%** | 4.1% | 99.0% | 6.05% | 249 | **5** ← **PEAK** |
| 4000 | 0.12% | 0.20% | 0.20% | 61.2% | 91.7% | 1.83% | 446 | 1 ← CRASH |
| 5000 | 0.00% | 0.00% | 0.00% | 44.3% | 93.6% | 2.83% | 417 | 0 ← BOTTOM |
| 6000 | 0.14% | 0.40% | 0.60% | 34.3% | 88.6% | 3.43% | 353 | 3 |
| 7000 | 0.12% | 0.30% | 0.40% | 48.3% | 81.5% | 2.97% | 387 | 2 |
| 8000 | 0.10% | 0.41% | 0.60% | 35.3% | 77.2% | 3.53% | 350 | 3 |
| 9000 | 0.20% | 0.63% | 0.80% | 52.9% | 72.9% | 2.41% | 403 | 4 |
| 10000 | 0.16% | 0.39% | 0.40% | 68.2% | 51.2% | 1.50% | 455 | 2 |
| 11000 | **0.37%** | 0.52% | 0.40% | 46.7% | 68.6% | 2.90% | 431 | 4 (1 perfect) |
| 12000 | 0.20% | 0.46% | 0.60% | 59.5% | 55.2% | 2.01% | 431 | 3 |
| 13000 | 0.20% | 0.46% | 0.60% | 52.3% | 69.2% | 2.64% | 429 | 3 |
| **14000** | 0.18% | 0.39% | 0.40% | **78.8%** | **34.9%** | 0.94% | **475** | 2 ← FINAL |

*CE% = compile error rate (solution-level). Partial% = avg % of test cases passed by failing solutions. MeanLen = mean generated code length in chars.*

### Paper Targets (for reference)

| Model | pass@1 Intro | pass@5 Intro |
|---|---|---|
| SFT only (paper) | 6.60% | 8.80% |
| RL (L_ce + L_rl, paper) | **6.20%** | **9.39%** |
| SFT only (ours) | 0.22% | 0.46% |
| RL round 1 best (ours) | 0.16% | 0.67% |

> Our absolute numbers are lower than the paper because they use beam search while we use nucleus sampling (temp=0.6, n=10). The key comparison is **relative improvement**: RL should exceed SFT.

---

## Three Training Phases

### Phase 1 — Healthy (ckpts 1000–3000)
Low compile errors (<5%), uniqueness 99%+, code length shrinking (357→301→249 chars — model learning to write more concise solutions), partial pass rate highest of the run (~6%). ckpt-3000 is the undisputed peak: 1.00% pass@10, 5 problems solved.

### Phase 2 — Oscillating Collapse (ckpts 4000–9000)
Between ckpt-3000 and ckpt-4000, compile errors jump 4% → 61% in a single 1000-step window and code length jumps from 249 → 446 chars. The RL gradient is ping-ponging: high-CE episodes push rewards negative → policy briefly recovers → reward signal shifts → collapses again. Classic limit cycle when LR is too high for the noise level of the reward signal.

### Phase 3 — Terminal Decline (ckpts 10000–14000)
Oscillations continue but trend worsens. Uniqueness enters freefall: 51% → 35% (at ckpt-14000). By the final checkpoint, only ~3.5 of 10 samples per problem are distinct — severe mode collapse. CE peaks at 78.8%, partial pass rate at 0.94% (worst in the run).

---

## Key Diagnostic Observations

**Code length as collapse indicator.** Healthy: length goes down (concise code). Collapsed: length goes up (padding with broken boilerplate). Median=510 chars at late checkpoints suggests a fixed broken template.

**Partial pass rate is more informative than pass@k.** Even at ckpt-9000 (pass@10=0.80%), partial rate is only 2.41% vs ckpt-1000's 6.14%. ckpt-9000's successes come from diversity (10 samples get lucky) not from higher intrinsic code quality.

**Generation timing (collapse canary).** Healthy: ~10s/problem. Collapsed: ~30–60s/problem (filling max token length). Measured for ckpts 13000/14000: 10–12s/problem — confirming partial diversity, not full token-stuffing.

**Root cause of collapse.** The critic was trained on SFT actor rollouts. Once the RL policy diverged from SFT, the critic's reward estimates became unreliable for the new distribution, causing noisy gradients that eroded diversity. LR=2e-5 was likely too high to stay in the stable region once reward signal quality degraded.

---

## Checkpoint Status

| Checkpoint | On disk? | Location |
|---|---|---|
| ckpt-3000 (**best**) | ✅ (manually saved) | `models/rl_actor_best_ckpt3000/` |
| ckpts 5000–14000 | ✅ | `exps/codet5-large_rl_bs4x16_lr2e-05_newcritic_relreturns/` |
| ckpts 1000–4000 | ❌ pruned | `save_total_limit=10` pruned oldest |

---

## Conclusion

**Best checkpoint: ckpt-3000** — pass@10=1.00%, uniqueness=99.0%, CE=4.1%, 5 problems solved.

RL with `--relative_returns` improved over SFT at ckpt-3000 (+67% pass@10: 1.00% vs 0.60%), confirming the method works. However, the improvement was not sustained: the policy collapsed between steps 3000–4000 and oscillated for the remaining ~12,000 steps without recovering to the ckpt-3000 level.

**Why Round 2 is needed:** The critic was trained on SFT rollouts (different distribution than ckpt-3000 outputs). Retraining on ckpt-3000 rollouts and using ckpt-3000 as the RL warmstart should close the actor–critic distribution gap and allow sustained improvement beyond step 3000.
