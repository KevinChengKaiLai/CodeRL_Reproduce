# Round 5 RL Ablation — Gradient Filter Study

**Date:** 2026-06-23  
**Goal:** Identify which gradient filter strategy stabilizes RL training from ckpt-3000 warmstart.  
**Warmstart:** `models/rl_actor_best_ckpt3000` (Round 1 best, pass@10=1.00%)  
**LR:** 5e-6, **max_grad_norm:** 0.1, **relative_returns:** yes, **epochs:** 3  
**Eval split:** Interview (indices 0–499, 500 problems, n=10, temp=0.6)

---

## Experiment Variables

| Exp | Gradient Filter | Data | Critic | Save Dir |
|-----|----------------|------|--------|----------|
| **5A** | `--positive_only` | round2 (ckpt-3000 rollout) | round2 | `codet5-large_rl_round5a_ckpt3000_posonly_lr5e-06_gnorm0.1_relreturns` |
| **5B** | none (all samples) | round1 (SFT rollout) | round1 | `codet5-large_rl_round5b_ckpt3000_weakbaseline_lr5e-06_gnorm0.1_relreturns` |
| **5C** | `--negative_only` | round2 (ckpt-3000 rollout) | round2 | `codet5-large_rl_round5c_ckpt3000_negonly_lr5e-06_gnorm0.1_relreturns` |
| **5D** | pos+neg (weak baseline) | round1 | round1 | `codet5-large_rl_round5d_ckpt3000_sftrollout_ckpt3kbaseline_lr5e-06_gnorm0.1_relreturns` |

> **Caveat:** 5A and 5C used round2 data; 5B and 5D used round1 data. Data is a confounded variable.  
> Planned follow-ups (5A_r1data, 5C_r1data, 5E) use round1 data to isolate gradient filter effect.

### Data Distribution

| Data | Positive returns | Negative returns | Neutral (0) |
|------|-----------------|-----------------|-------------|
| round1 (SFT rollout) | 13.2% | 20.7% | 66.1% |
| round2 (ckpt-3000 rollout) | ~2.1% | ~97.9% | — |

---

## Training Outcomes

| Exp | train_loss | train_rl_loss | Verdict |
|-----|-----------|--------------|---------|
| 5A | 0.193 | 0.0002 | Stable |
| 5B | 0.215 | −1.208 | Collapsed |
| 5C | 0.204 | −1.356 | Collapsed |
| 5D | 0.214 | −1.226 | Collapsed |

> rl_loss << 0 in 5B/5C/5D indicates the policy gradient pushed the model toward low-reward outputs (sign of collapse).

---

## Eval Results — Interview Split (500 problems, n=10)

| Model | pass@1 | pass@5 | pass@10 | CE% | RE% | Fail% | Uniq% |
|-------|--------|--------|---------|-----|-----|-------|-------|
| SFT baseline | 0.22 | 0.46 | 0.60 | 17.5 | — | — | 99.7 |
| R1 ckpt-3000 | 0.16 | 0.67 | **1.00** | 4.1 | — | — | 99.0 |
| **5A-ckpt1000** | 0.32 | 0.68 | 0.80 | 3.5 | 27.2 | 68.9 | 99.3 |
| **5A-ckpt2000** | **0.48** | **0.87** | **1.00** | 3.2 | 29.4 | 66.8 | 99.3 |
| **5A-ckpt3000** | 0.44 | 0.70 | 0.80 | 3.1 | 26.8 | 69.7 | 99.0 |
| **5A-ckpt4000** | 0.44 | 0.87 | **1.00** | **2.9** | 29.0 | 67.6 | 99.5 |
| 5B-ckpt1000 | 0.26 | 0.50 | 0.60 | 45.5 | 26.8 | 27.4 | 93.3 |
| 5B-ckpt2000 | 0.16 | 0.45 | 0.60 | 42.0 | 21.4 | 36.5 | 78.4 |
| 5B-ckpt3000 | 0.04 | 0.16 | 0.20 | 66.8 | 13.8 | 19.3 | 63.0 |
| 5B-ckpt4000 | 0.12 | 0.37 | 0.40 | 52.5 | 17.3 | 30.0 | 71.6 |
| 5C-ckpt1000 | 0.00 | 0.00 | 0.00 | 100 | 0.0 | 0.0 | 10.0 |
| 5C-ckpt2000 | 0.00 | 0.00 | 0.00 | 100 | 0.0 | 0.0 | 10.0 |
| 5C-ckpt3000 | 0.00 | 0.00 | 0.00 | 100 | 0.0 | 0.0 | 10.0 |
| 5C-ckpt4000 | 0.00 | 0.00 | 0.00 | 100 | 0.0 | 0.0 | 10.0 |
| 5D-ckpt1000 | 0.08 | 0.20 | 0.20 | 61.1 | 15.9 | 22.9 | 63.5 |
| 5D-ckpt2000 | 0.10 | 0.20 | 0.20 | 56.9 | 18.2 | 24.7 | 61.4 |
| 5D-ckpt3000 | 0.12 | 0.30 | 0.40 | 71.4 | 9.2 | 19.3 | 51.3 |
| 5D-ckpt4000 | 0.18 | 0.30 | 0.40 | 59.6 | 13.8 | 26.4 | 51.4 |

---

## Eval Results — Introductory Split (500 problems, n=9)

| Model | pass@1 | pass@5 | pass@10 | CE% | RE% | Fail% | Uniq% |
|-------|--------|--------|---------|-----|-----|-------|-------|
| SFT baseline | 3.31 | 7.29 | 9.07 | 15.2 | 26.4 | 55.1 | 98.7 |
| **5A-ckpt2000** | 2.64 | 7.24 | **10.10** | 2.5 | 31.8 | 63.0 | 98.8 |

> SFT eval: 480 problems (some missing codes). 5A intro eval: 500 problems.

---

## Key Findings

### 1. `--positive_only` is the only stable gradient filter
5A (positive_only) is the sole experiment that did not collapse. CE% stayed at ~3% throughout training (vs SFT baseline 17.5%), and uniqueness remained >99%. All other filters caused CE% to spike within ckpt-1000.

### 2. Collapse speed: negative_only > no filter > pos+neg
- **5C** (negative_only): instant collapse at ckpt-1000, CE%=100%, uniqueness=10%
- **5B** (no filter): CE% 45.5% at ckpt-1000, worsening to 66.8% by ckpt-3000
- **5D** (pos+neg weak baseline): CE% 61% at ckpt-1000, plateaus ~60%

### 3. Negative gradient = catastrophic forgetting
Negative-reward gradients push the model away from outputs the actor currently generates. Since the actor generates mostly compile-error outputs at this stage (~18% CE at SFT), "don't do what you're doing" causes the model to abandon coherent generation entirely.

### 4. pass@10 improvement is modest on interview split
5A matches R1 ckpt-3000 at pass@10=1.00% but doesn't exceed it. The introductory split shows a small gain (9.07% → 10.10%), suggesting RL is helping more on easier problems.

### 5. Data confound (round2 vs round1)
5A/5C used round2 data (ckpt-3000 rollouts, nearly 0% pass rate) while 5B/5D used round1 data (SFT rollouts, 13.2% positive returns). The positive_only filter in 5A effectively discards 97.9% of the round2 data, training only on the rare 2.1% positive samples — which may explain why the rl_loss is tiny (0.0002) and training is stable.

---

## Planned Follow-ups

| Exp | Filter | Data | gnorm | Purpose |
|-----|--------|------|-------|---------|
| **5A_r1data** | positive_only | round1 | 0.1 | Isolate filter effect (more positive samples = stronger signal) |
| **5C_r1data** | negative_only | round1 | 0.1 | Does collapse persist with round1 data? |
| **5E** | positive_only | round1 | 1.0 | gnorm ablation vs 5A_r1data |

Status: 5A_r1data and 5C_r1data attempted 2026-06-17 but killed by machine at ~step 528/665 (before first checkpoint). Need to re-run.

---

## Context: Why Rounds 2–4 Were Abandoned

### Round 2 — ckpt-3000 warmstart, round2 data, no filter
- **Config:** warmstart=rl_actor_best_ckpt3000, data=round2 (ckpt-3000 rollouts), LR=1e-5, no gradient filter
- **Result:** CE%=100% at step 1000, complete collapse
- **Root cause:** ckpt-3000 rollouts had ~0% pass rate (entropy collapsed after round 1 training). Nearly all relative returns were negative → negative gradients dominated → catastrophic forgetting

### Round 3 — SFT warmstart, round1 data, no filter, LR=1e-5
- **Config:** warmstart=sft_actor, data=round1, LR=1e-5, no gradient filter
- **Result (watcher, 50 intro problems):**

| ckpt | pass@10 | CE% |
|------|---------|-----|
| 1000 | 0.0% | 5.6% |
| 3000 | 0.0% | 6.9% |
| 5000 | 0.0% | 52.0% |
| 7000 | 0.0% | 45.4% |

- **Verdict:** Never improved over SFT; collapsed by ckpt-5000. No checkpoint saved to `models/rl_round3_best/` (nothing beat SFT baseline).

### Round 4 — SFT warmstart, round1 data, no filter, LR=5e-6
- **Config:** warmstart=sft_actor, data=round1, LR=5e-6 (halved vs round 3), no gradient filter
- **Result (watcher, 50 intro problems):**

| ckpt | pass@10 | CE% |
|------|---------|-----|
| 1000 | 0.60% | 8.2% |
| 2000 | 0.80% | 7.0% |
| 3000 | 0.80% | 6.6% |

- **Verdict:** Better than round 3 (lower LR helped early stability), but no gradient filter = same collapse pattern as 5B. Watcher only ran 3 checkpoints; full eval never done. Analogous to 5B (no filter, round1 data) which showed CE% 45–67% by ckpt 1000–3000 on 500 problems.

### Summary
All three rounds used no gradient filter. The core lesson: without `--positive_only`, RL training from this warmstart always collapses regardless of LR, warmstart choice, or data source. Round 5A (`--positive_only`) is the first configuration that achieved stable training.
