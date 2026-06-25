#!/usr/bin/env python3
"""Visualize all eval metrics across training checkpoints."""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# ─── Data (parsed from all eval_summary.txt files) ───────────────────────────
# Excluded: outputs/results_rl/ — no step number in directory name.
# Excluded duplicate: rl_redo_result/rl_newcritic_ckpt7000 (identical to test_results_rl_redo_ckpt_7000).
#
# Fields per record:
#   step, pass1, pass5, pass10, partial_pass,
#   compile%, runtime%, failed%, passed%,
#   code_len_mean, uniqueness%

SFT = dict(
    step=0, group="SFT", label="SFT",
    pass1=0.22, pass5=0.46, pass10=0.60, partial_pass=5.53,
    compile=17.5, runtime=24.2, failed=58.1, passed=0.2,
    code_len=381.5, uniqueness=99.7,
)

# Old RL run (no --relative_returns, defaulted to codet5-base backbone)
RL_V1_RAW = [
    #  step   p1     p5     p10   pp      ce     re     ft     pt     cl       un
    (  1000, 0.14,  0.20,  0.20,  2.24,  61.4,  12.9,  25.6,  0.1,  1347.3,  99.3),
    (  7000, 0.00,  0.00,  0.00,  0.01,  99.5,   0.1,   0.4,  0.0,  6383.4,   5.7),
    ( 20000, 0.28,  0.40,  0.40,  0.09,  99.4,   0.1,   0.2,  0.3,  6561.1,  10.6),
]

# New RL run (new critic weights, still no --relative_returns)
RL_V2_RAW = [
    ( 21000, 0.38,  0.40,  0.40,  0.15,  98.2,   0.3,   1.0,  0.4,  6499.4,  11.6),
    ( 22000, 0.20,  0.30,  0.40,  0.00,  99.2,   0.1,   0.4,  0.2,  6569.9,  10.8),
    ( 23000, 0.20,  0.20,  0.20,  0.16,  99.2,   0.0,   0.5,  0.2,  6572.4,  10.7),
    ( 24000, 0.20,  0.38,  0.40,  0.08,  99.4,   0.1,   0.3,  0.2,  6589.4,  10.4),
    ( 25000, 0.10,  0.20,  0.20,  0.13,  99.1,   0.1,   0.7,  0.1,  6542.3,  12.5),
    ( 26000, 0.00,  0.00,  0.00,  0.12,  99.6,   0.1,   0.3,  0.0,  6594.8,  10.6),
    ( 27000, 0.16,  0.30,  0.40,  0.11,  99.3,   0.2,   0.3,  0.2,  6552.1,  10.9),
    ( 28000, 0.18,  0.20,  0.20,  0.07,  99.5,   0.0,   0.3,  0.2,  6499.9,  11.1),
    ( 29000, 0.00,  0.00,  0.00,  0.08,  99.7,   0.0,   0.2,  0.0,  6539.2,  10.6),
    ( 30000, 0.20,  0.20,  0.20,  0.13,  99.3,   0.0,   0.5,  0.2,  6563.7,  10.7),
]

KEYS = ["step", "pass1", "pass5", "pass10", "partial_pass",
        "compile", "runtime", "failed", "passed", "code_len", "uniqueness"]

def make_records(raw, group):
    return [dict(zip(KEYS, row), group=group) for row in raw]

v1 = make_records(RL_V1_RAW, "RL-v1")
v2 = make_records(RL_V2_RAW, "RL-v2")

v1_steps = [r["step"] for r in v1]
v2_steps = [r["step"] for r in v2]

# ─── Colors / style ──────────────────────────────────────────────────────────
C_SFT = "#2ca02c"
C_V1  = "#ff7f0e"
C_V2  = "#1f77b4"
C_CE  = "#d62728"
C_RE  = "#ff7f0e"
C_FT  = "#bcbd22"
C_PT  = "#2ca02c"

# ─── Figure layout ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(21, 17))
fig.patch.set_facecolor("#f5f5f5")
fig.suptitle(
    "CodeRL Reproduction — Eval Metrics Across Training Checkpoints\n"
    "APPS Introductory (500 problems) · n=10 samples/problem · temp=0.6\n"
    "Green dashed = SFT baseline  |  Orange ▲ = RL-v1 (old run, absolute returns)  |  "
    "Blue ● = RL-v2 (new critic, absolute returns)",
    fontsize=11, fontweight="bold", y=0.995, va="top",
)

gs = GridSpec(3, 3, figure=fig,
              hspace=0.55, wspace=0.38,
              left=0.07, right=0.97, top=0.935, bottom=0.06)

# ─── Shared helpers ──────────────────────────────────────────────────────────
def style_ax(ax, title, ylabel="", xlabel="Training Step"):
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.28, linestyle="--", linewidth=0.6)
    ax.tick_params(labelsize=7.5)
    ax.set_facecolor("#fafafa")
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)

def sft_hline(ax, key, fmt="{:.2f}%"):
    val = SFT[key]
    ax.axhline(val, color=C_SFT, linewidth=2, linestyle="--", alpha=0.9, zorder=3,
               label=f"SFT ({fmt.format(val)})")

def v1_scatter(ax, key):
    vals = [r[key] for r in v1]
    ax.scatter(v1_steps, vals, color=C_V1, marker="^", s=75, zorder=7,
               edgecolors="white", linewidths=0.6, label="RL-v1 (old run)")

def v2_line(ax, key):
    vals = [r[key] for r in v2]
    ax.plot(v2_steps, vals, color=C_V2, marker="o", markersize=5.5,
            linewidth=2, zorder=6, label="RL-v2 (new critic)")
    ax.fill_between(v2_steps, vals, alpha=0.07, color=C_V2)

def add_legend(ax, loc="best"):
    ax.legend(fontsize=7, loc=loc, framealpha=0.85, edgecolor="#cccccc")

# ─── Row 0: pass@k ───────────────────────────────────────────────────────────
for col, key, title in [
    (0, "pass1",  "pass@1 (%)"),
    (1, "pass5",  "pass@5 (%)"),
    (2, "pass10", "pass@10 (%)"),
]:
    ax = fig.add_subplot(gs[0, col])
    sft_hline(ax, key)
    v1_scatter(ax, key)
    v2_line(ax, key)
    add_legend(ax, loc="upper right")
    style_ax(ax, title, ylabel=title)
    # mark best RL-v2 checkpoint
    best_idx = np.argmax([r[key] for r in v2])
    best_step = v2_steps[best_idx]
    best_val  = v2[best_idx][key]
    ax.annotate(f"best\n{best_val:.2f}%",
                xy=(best_step, best_val),
                xytext=(best_step - 2000, best_val + 0.04),
                fontsize=6.5, color=C_V2,
                arrowprops=dict(arrowstyle="->", color=C_V2, lw=0.8))

# ─── Row 1: Compile error, Uniqueness, Code length ───────────────────────────
for col, key, title, ylabel, fmt in [
    (0, "compile",    "Compile Error Rate (%)",        "CompileError (%)",    "{:.1f}%"),
    (1, "uniqueness", "Sample Uniqueness (%)",          "Uniqueness (%)",      "{:.1f}%"),
    (2, "code_len",   "Mean Generated Code Length",     "Mean chars",          "{:.0f} chars"),
]:
    ax = fig.add_subplot(gs[1, col])
    sft_hline(ax, key, fmt=fmt)
    v1_scatter(ax, key)
    v2_line(ax, key)
    add_legend(ax)
    style_ax(ax, title, ylabel=ylabel)

# ─── Row 2 col 0: Partial Test Pass Rate ─────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
sft_hline(ax7, "partial_pass")
v1_scatter(ax7, "partial_pass")
v2_line(ax7, "partial_pass")
add_legend(ax7)
style_ax(ax7,
         "Partial Test Pass Rate (%)\n(avg % of test cases passed per failing solution)",
         ylabel="Partial pass (%)")

# ─── Row 2 cols 1-2: Stacked error type bar (all checkpoints) ────────────────
ax8 = fig.add_subplot(gs[2, 1:])

bar_labels = (
    ["SFT"]
    + [f"RL-v1\n{s//1000}k" for s in v1_steps]
    + [f"RL-v2\n{s//1000}k" for s in v2_steps]
)
all_recs = [SFT] + v1 + v2
ce_vals = [r["compile"]  for r in all_recs]
re_vals = [r["runtime"]  for r in all_recs]
ft_vals = [r["failed"]   for r in all_recs]
pt_vals = [r["passed"]   for r in all_recs]

x = np.arange(len(bar_labels))
w = 0.65

bot = np.zeros(len(x))
ax8.bar(x, ce_vals, w, bottom=bot, label="CompileError", color=C_CE)
bot += np.array(ce_vals)
ax8.bar(x, re_vals, w, bottom=bot, label="RuntimeError", color=C_RE)
bot += np.array(re_vals)
ax8.bar(x, ft_vals, w, bottom=bot, label="FailedTest",   color=C_FT)
bot += np.array(ft_vals)
ax8.bar(x, pt_vals, w, bottom=bot, label="PassedTest",   color=C_PT)

# shade SFT and v1 region backgrounds
ax8.axvspan(-0.5,  0.5, alpha=0.07, color=C_SFT)
ax8.axvspan( 0.5,  3.5, alpha=0.07, color=C_V1)
ax8.axvspan( 3.5, 13.5, alpha=0.07, color=C_V2)

ax8.text(0,   102, "SFT",   ha="center", fontsize=7, color=C_SFT, fontweight="bold")
ax8.text(2,   102, "RL-v1", ha="center", fontsize=7, color=C_V1,  fontweight="bold")
ax8.text(8.5, 102, "RL-v2 (new critic checkpoints)",
         ha="center", fontsize=7, color=C_V2, fontweight="bold")

ax8.set_xticks(x)
ax8.set_xticklabels(bar_labels, fontsize=6.5)
ax8.set_ylim(0, 107)
ax8.legend(fontsize=7.5, ncol=4, loc="lower right", framealpha=0.9)
style_ax(ax8,
         "Solution-level Error Type Distribution (% of all solutions)",
         xlabel="Checkpoint", ylabel="% of solutions")

# ─── Save ────────────────────────────────────────────────────────────────────
out_path = "outputs/metrics_overview.jpg"
os.makedirs("outputs", exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor(), format="jpeg")
print(f"Saved → {out_path}")
plt.close(fig)
