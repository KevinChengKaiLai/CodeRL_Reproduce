#!/usr/bin/env python3
"""
Diagnose raw gradient norm vs reward group for rl_actor_best_ckpt3000.

Runs 200 RL forward+backward passes (NO optimizer.step) and records
(relative_reward, grad_norm) per batch. Plots:
  - Box plot: grad norm by reward sign (positive / zero / negative)
  - Scatter: relative reward vs grad norm
Both on log-scale y-axis. Outputs grad_norm_log.csv and grad_norm_analysis.png.
"""

import os
import csv
import random
import numpy as np
import torch
import torch.multiprocessing
import transformers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.apps_dataset import APPSBaseDataset

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH  = 'models/rl_actor_best_ckpt3000'
TRAIN_PATH  = 'data/APPS/train/'
NUM_BATCHES = 200
BATCH_SIZE  = 2          # matches round2_train_rl.sh
DEVICE      = 'cuda:0'
SEED        = 42
LOG_CSV     = 'grad_norm_log.csv'
PLOT_PNG    = 'grad_norm_analysis.png'

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.multiprocessing.set_sharing_strategy('file_system')


# ── Collate fn ─────────────────────────────────────────────────────────────────
def collate_fn(batch):
    """Stack fixed-size tensors; pad variable-length tensors to batch max."""
    out = {}
    for key in batch[0]:
        tensors = [item[key] for item in batch]
        if all(t.shape == tensors[0].shape for t in tensors):
            out[key] = torch.stack(tensors)
        else:
            max_len = max(t.shape[0] for t in tensors)
            pad_val = -100 if 'label' in key else 0
            padded = torch.full(
                (len(tensors), max_len), pad_val, dtype=tensors[0].dtype
            )
            for i, t in enumerate(tensors):
                padded[i, :t.shape[0]] = t
            out[key] = padded
    return out


# ── Load model ─────────────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH} ...")
model = transformers.T5ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    tuning_mode='rl',
    clone_rl_head=False,
)
model.to(DEVICE)
model.train()
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model loaded: {n_params:.1f}M params")


# ── Load dataset ───────────────────────────────────────────────────────────────
print(f"Loading dataset from {TRAIN_PATH} ...")
problem_dirs = sorted(os.listdir(TRAIN_PATH))
dataset = APPSBaseDataset(
    dataroot=TRAIN_PATH,
    problem_dirs=problem_dirs,
    model='codet5-large',
    max_tokens=512,
    max_src_tokens=600,
    sample_mode='uniform_sol',
    tuning_mode='rl',
    relative_returns=True,
)
print(f"Dataset size: {len(dataset)} RL samples")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)


# ── Diagnostic loop ────────────────────────────────────────────────────────────
records = []   # list of (relative_reward: float, grad_norm: float)

print(f"\nRunning {NUM_BATCHES} RL forward+backward passes (no optimizer.step) ...")
data_iter = iter(loader)

for i in range(NUM_BATCHES):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        batch = next(data_iter)

    rl_input_ids  = batch['rl_input_ids'].to(DEVICE)
    rl_rewards    = batch['rl_rewards'].to(DEVICE)
    rl_label_ids  = batch['rl_label_ids'].to(DEVICE)

    # relative_reward for this batch: mean of non-zero reward values
    non_zero = rl_rewards[rl_rewards != 0.0]
    relative_reward = non_zero.mean().item() if len(non_zero) > 0 else 0.0

    # RL forward + backward (mirrors compute_loss at odd step in trainer_rl.py)
    model.zero_grad()
    rl_loss = model(
        input_ids=rl_input_ids,
        rewards=rl_rewards,
        labels=rl_label_ids,
    )
    rl_loss.backward()

    # Raw gradient norm before any clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float('inf')
    ).item()

    records.append((relative_reward, grad_norm))

    if (i + 1) % 20 == 0:
        sign = '+' if relative_reward > 0 else ('0' if relative_reward == 0 else '-')
        print(
            f"  [{i+1:3d}/{NUM_BATCHES}]  "
            f"rel_reward={relative_reward:+.4f} ({sign})  "
            f"grad_norm={grad_norm:.2f}"
        )


# ── Save CSV ───────────────────────────────────────────────────────────────────
with open(LOG_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['relative_reward', 'grad_norm'])
    writer.writerows(records)
print(f"\nSaved {len(records)} records → {LOG_CSV}")


# ── Plot ───────────────────────────────────────────────────────────────────────
rewards_arr = np.array([r[0] for r in records])
norms_arr   = np.array([r[1] for r in records])

pos_mask  = rewards_arr > 0
zero_mask = rewards_arr == 0.0
neg_mask  = rewards_arr < 0

groups = []
labels = []
for mask, sign_label in [(pos_mask, 'Positive'), (zero_mask, 'Zero'), (neg_mask, 'Negative')]:
    if mask.any():
        groups.append(norms_arr[mask])
        labels.append(f'{sign_label}\n(n={mask.sum()})')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: box plot
ax1.boxplot(groups, labels=labels, patch_artist=True,
            boxprops=dict(facecolor='lightsteelblue', alpha=0.7))
ax1.set_yscale('log')
ax1.set_ylabel('Gradient Norm (log scale)')
ax1.set_title('Gradient Norm by Reward Sign')
ax1.grid(True, which='both', alpha=0.3)

# Right: scatter plot
colors = np.where(pos_mask, 'green', np.where(neg_mask, 'red', 'gray'))
ax2.scatter(rewards_arr, norms_arr, c=colors, alpha=0.6, s=30, edgecolors='none')
ax2.set_yscale('log')
ax2.set_xlabel('Relative Reward (batch mean of non-zero rl_rewards)')
ax2.set_ylabel('Gradient Norm (log scale)')
ax2.set_title('Relative Reward vs Gradient Norm')
ax2.grid(True, which='both', alpha=0.3)
# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Positive reward'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',   markersize=8, label='Negative reward'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',  markersize=8, label='Zero reward'),
]
ax2.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(PLOT_PNG, dpi=150, bbox_inches='tight')
print(f"Saved plot → {PLOT_PNG}")

# ── Summary stats ──────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────")
for mask, label in [(pos_mask, 'Positive'), (zero_mask, 'Zero'), (neg_mask, 'Negative')]:
    if mask.any():
        g = norms_arr[mask]
        print(f"  {label:8s}  n={mask.sum():4d}  "
              f"median={np.median(g):.2f}  "
              f"mean={np.mean(g):.2f}  "
              f"max={np.max(g):.2f}")
print("─────────────────────────────────────────────────────────")
