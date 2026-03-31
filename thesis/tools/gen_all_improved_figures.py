#!/usr/bin/env python3
"""
Generate ALL improved thesis figures using publication-quality templates.

Produces (in thesis/figures/):
  1. planner_comparison_v2.png    — grouped bar with correct data + error bars placeholder
  2. seg_improvement_v2.png       — 4-panel bar chart with correct data
  3. runtime_breakdown_v2.png     — dual-pipeline stacked horizontal bars
  4. training_curves_v2.png       — loss + IoU curves from actual training JSON
  5. bev_fragility_v2.png         — corrected BEV failure rate chart
  6. latency_log_scale.png        — log-scale latency comparison (new)
  7. planner_speedup_ratio.png    — speedup ratios vs BEV baseline (new)
  8. iteration_timeline.png       — design iteration progression (new)

Usage:
    python thesis/tools/gen_all_improved_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "thesis" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Add thesis tools to path for style utilities
sys.path.insert(0, str(ROOT / "thesis" / "tools"))
from thesis_figure_utils import (
    apply_thesis_style, save_thesis_figure,
    BEV_COLOR, IMG_COLOR, TEMPLATE_COLOR, BASELINE_COLOR, COLORS,
)

apply_thesis_style()


# ═══════════════════════════════════════════════════════════════════════════
# DATA FROM THESIS TABLES (verified against main.tex Tables 5.1, 5.3, etc.)
# ═══════════════════════════════════════════════════════════════════════════

# Table 5.3: Planner comparison on 32 hand-annotated frames
PLANNER_DATA = {
    "names":      ["BEV\nSkeleton", "BEV\nDT Ridge", "BEV DT\n(near)", "Img\nMidpoint", "Img\nDT"],
    "center_err": [76.6, 65.0, 79.0, 14.3, 60.4],
    "inside_gt":  [0.971, 0.986, 0.986, 0.985, 0.994],
    "latency_ms": [380.3, 926.8, 1603.0, 2.2, 108.1],
    "domain":     ["bev", "bev", "bev", "img", "img"],
}

# Table 5.1: Segmentation comparison on 32 hand-annotated frames
SEG_DATA = {
    "models": ["Baseline\n(B2 teacher)", "Candidate\n(OneFormer teacher)", "Candidate\n+ confhold"],
    "iou":       [0.758, 0.946, 0.903],
    "precision": [0.910, 0.983, 0.987],
    "recall":    [0.835, 0.962, 0.914],
    "f1":        [0.851, 0.972, 0.947],
    "ms":        [18.9, 11.7, 13.1],
}

# Table 5.9: Runtime comparison BEV vs Image-space pipeline
RUNTIME_BEV = {"Segmentation": 11.7, "Mask Refine": 8.5, "BEV Proj": 0.9, "BEV Cleanup": 14.6, "Planner": 380.3}
RUNTIME_IMG = {"Segmentation": 11.7, "Mask Refine": 3.0, "Planner": 2.2}

# Table 5.6: BEV fragility (baseline model, 4407 frames)
BEV_FRAGILITY = {"valid (dt_ridge)": 10, "held (dt_ridge_hold)": 20, "no path (none)": 4377}

# Table 5.2: Full-video replay (22,679 frames)
FULLVIDEO = {
    "metric":          ["Mean Seg IoU", "Unstable Rate (%)", "Template Success (%)", "Fallback Rate (%)"],
    "baseline":        [0.909, 1.46, 73.7, 19.0],
    "candidate":       [0.925, 0.33, 79.3, 14.3],
}

# Table 5.8: Design iteration progression
ITERATIONS = {
    "version": ["v1", "v2", "v3", "v4"],
    "planner": ["Skeleton-Graph", "DT Ridge", "Midpoint / DT", "Template Arc"],
    "domain":  ["BEV", "BEV", "Image-space", "BEV corridor"],
    "fps":     [9.1, 1.0, 59.2, 9.97],
}


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Planner Comparison (corrected data from Table 5.3)
# ═══════════════════════════════════════════════════════════════════════════

def gen_planner_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = PLANNER_DATA["names"]
    n = len(names)
    x = np.arange(n)
    colors = [BEV_COLOR if d == "bev" else IMG_COLOR for d in PLANNER_DATA["domain"]]

    # Panel 1: Center Error
    ax = axes[0]
    bars = ax.bar(x, PLANNER_DATA["center_err"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Lateral Center Error (px)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    for bar, val in zip(bars, PLANNER_DATA["center_err"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    # Panel 2: Inside-GT
    ax = axes[1]
    bars = ax.bar(x, [v * 100 for v in PLANNER_DATA["inside_gt"]], color=colors,
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Inside-GT (%)")
    ax.set_ylim(95, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    for bar, val in zip(bars, PLANNER_DATA["inside_gt"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    # Panel 3: Latency (log scale)
    ax = axes[2]
    bars = ax.bar(x, PLANNER_DATA["latency_ms"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.axhline(y=100, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="100 ms budget")
    ax.legend(fontsize=7)
    for bar, val in zip(bars, PLANNER_DATA["latency_ms"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                f"{val:.1f}", ha="center", va="bottom", fontsize=6)

    # Legend
    bev_patch = mpatches.Patch(color=BEV_COLOR, label="BEV-domain")
    img_patch = mpatches.Patch(color=IMG_COLOR, label="Image-space")
    fig.legend(handles=[bev_patch, img_patch], loc="upper center", ncol=2,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Planner Comparison — 32 Hand-Annotated Frames", y=1.06, fontsize=11)
    save_thesis_figure(fig, "planner_comparison_v2", output_dir=FIGURES)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Segmentation Improvement (corrected data from Table 5.1)
# ═══════════════════════════════════════════════════════════════════════════

def gen_seg_improvement():
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    models = SEG_DATA["models"]
    x = np.arange(len(models))
    bar_colors = [BASELINE_COLOR, TEMPLATE_COLOR, COLORS["orange"]]

    panels = [
        ("IoU", SEG_DATA["iou"], (0.6, 1.0)),
        ("Precision", SEG_DATA["precision"], (0.8, 1.0)),
        ("Recall", SEG_DATA["recall"], (0.8, 1.0)),
        ("Inference (ms)", SEG_DATA["ms"], None),
    ]

    for ax, (label, values, ylim) in zip(axes, panels):
        bars = ax.bar(x, values, color=bar_colors, edgecolor="black", linewidth=0.5, width=0.6)
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=7)
        if ylim:
            ax.set_ylim(ylim)
        for bar, val in zip(bars, values):
            fmt = f"{val:.3f}" if val < 2 else f"{val:.1f}"
            ypos = bar.get_height() + (0.005 if ylim else 0.3)
            ax.text(bar.get_x() + bar.get_width() / 2, ypos, fmt,
                    ha="center", va="bottom", fontsize=7)

    fig.suptitle("Segmentation Quality — 32 Hand-Annotated Frames", fontsize=11)
    save_thesis_figure(fig, "seg_improvement_v2", output_dir=FIGURES)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Runtime Breakdown — Dual Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def gen_runtime_breakdown():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [3, 2]})

    # ── Left panel: side-by-side grouped bars per module ──
    ax = axes[0]
    all_modules = ["Segmentation", "Mask Refine", "BEV Proj", "BEV Cleanup", "Planner"]
    bev_vals = [RUNTIME_BEV.get(m, 0) for m in all_modules]
    img_vals = [RUNTIME_IMG.get(m, 0) for m in all_modules]

    x = np.arange(len(all_modules))
    w = 0.35
    bars_bev = ax.bar(x - w / 2, bev_vals, w, color=BEV_COLOR, edgecolor="black",
                      linewidth=0.5, label="BEV Pipeline")
    bars_img = ax.bar(x + w / 2, img_vals, w, color=IMG_COLOR, edgecolor="black",
                      linewidth=0.5, label="Image Pipeline")

    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(all_modules, fontsize=8)
    ax.axhline(y=100, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=8)
    ax.set_title("Per-Module Latency")

    # Value labels
    for bar, val in zip(bars_bev, bev_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val * 1.2,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6.5)
    for bar, val in zip(bars_img, img_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, max(val * 1.2, 0.3),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6.5)

    # ── Right panel: total comparison ──
    ax2 = axes[1]
    bev_total = sum(RUNTIME_BEV.values())
    img_total = sum(RUNTIME_IMG.values())

    bars = ax2.barh(["Image\nPipeline", "BEV\nPipeline"],
                    [img_total, bev_total],
                    color=[IMG_COLOR, BEV_COLOR], edgecolor="black", linewidth=0.5,
                    height=0.5)
    ax2.axvline(x=100, color="red", linestyle="--", linewidth=1, alpha=0.7, label="100 ms budget")
    ax2.set_xlabel("Total Latency (ms)")
    ax2.set_title("Pipeline Total")

    ax2.text(img_total + 8, 0, f"{img_total:.1f} ms\n({1000/img_total:.0f} FPS)",
             va="center", fontsize=9, fontweight="bold", color=IMG_COLOR)
    ax2.text(bev_total + 8, 1, f"{bev_total:.0f} ms\n({1000/bev_total:.1f} FPS)",
             va="center", fontsize=9, fontweight="bold", color=BEV_COLOR)

    # Speedup annotation
    speedup = bev_total / img_total
    ax2.text(bev_total / 2, 0.5, f"{speedup:.0f}× faster",
             ha="center", va="center", fontsize=11, fontweight="bold",
             color="white", bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7))

    ax2.legend(fontsize=7)
    fig.suptitle("Runtime Comparison: BEV vs. Image-Space Pipeline ($640\\times360$, CPU)",
                 fontsize=11)
    save_thesis_figure(fig, "runtime_breakdown_v2", output_dir=FIGURES)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Training Curves from actual JSON
# ═══════════════════════════════════════════════════════════════════════════

def gen_training_curves():
    # Load training histories
    training_dir = ROOT / "outputs" / "training"
    runs = {}
    for run_dir in sorted(training_dir.iterdir()):
        summary = run_dir / "best_checkpoint" / "training_summary.json"
        if summary.exists():
            with open(summary) as f:
                data = json.load(f)
            if "history" in data and len(data["history"]) > 0:
                runs[run_dir.name] = data

    if not runs:
        print("No training histories found, skipping training curves.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Short display names
    name_map = {
        "binary_segformer_oneformer_teacher": "Stage 1 (OneFormer teacher)",
        "binary_segformer_all6_t400": "All 6 videos (scratch)",
        "binary_segformer_all6_t400_stage2": "Stage 2 (all 6, fine-tune)",
        "binary_segformer_old400_plus_img_1931_t300": "Stage 2 (+IMG_1931)",
    }

    palette = [COLORS["blue"], COLORS["vermillion"], COLORS["green"], COLORS["orange"]]

    for i, (key, data) in enumerate(runs.items()):
        history = data["history"]
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_iou = [h["val_iou"] for h in history]
        label = name_map.get(key, key)
        color = palette[i % len(palette)]

        axes[0].plot(epochs, train_loss, marker="o", markersize=4, color=color,
                     label=label, linewidth=1.5)
        axes[1].plot(epochs, val_iou, marker="s", markersize=4, color=color,
                     label=label, linewidth=1.5)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend(fontsize=7, loc="upper right")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation IoU")
    axes[1].set_title("Validation IoU")
    axes[1].set_ylim(0.88, 0.97)
    axes[1].legend(fontsize=7, loc="lower right")

    fig.suptitle("SegFormer-B0 Training Progression", fontsize=11)
    save_thesis_figure(fig, "training_curves_v2", output_dir=FIGURES)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: BEV Fragility (corrected — baseline model data)
# ═══════════════════════════════════════════════════════════════════════════

def gen_bev_fragility():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Pie chart of outcomes
    labels = list(BEV_FRAGILITY.keys())
    sizes = list(BEV_FRAGILITY.values())
    colors_pie = [TEMPLATE_COLOR, COLORS["orange"], BASELINE_COLOR]
    explode = (0.05, 0.05, 0)

    wedges, texts, autotexts = axes[0].pie(
        sizes, labels=None, autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
        colors=colors_pie, explode=explode, startangle=90,
        textprops={"fontsize": 9}, pctdistance=0.6,
    )
    axes[0].legend(wedges, [f"{l} ({v:,})" for l, v in zip(labels, sizes)],
                   loc="lower left", fontsize=7)
    axes[0].set_title("BEV Path Extraction Outcome\n(4,407 frames, baseline model)")

    # Right: Bar showing the dominance of failure
    categories = ["Valid\nPath", "Held\nPath", "No\nPath"]
    values = [10, 20, 4377]
    pcts = [v / sum(values) * 100 for v in values]
    bar_colors = [TEMPLATE_COLOR, COLORS["orange"], "#CC3333"]

    bars = axes[1].bar(categories, pcts, color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Frames (%)")
    axes[1].set_title("BEV Planning Failure Rate")
    for bar, val, pct in zip(bars, values, pcts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{pct:.1f}%\n(n={val:,})", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Monocular BEV Projection Fragility — Baseline Segmentation Model", fontsize=11)
    save_thesis_figure(fig, "bev_fragility_v2", output_dir=FIGURES)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Latency Log-Scale Comparison (NEW)
# ═══════════════════════════════════════════════════════════════════════════

def gen_latency_log_scale():
    fig, ax = plt.subplots(figsize=(7, 4))

    names = ["BEV\nSkeleton", "BEV\nDT Ridge", "BEV DT\n(near)", "Img\nMidpoint", "Img\nDT"]
    latencies = [380.3, 926.8, 1603.0, 2.2, 108.1]
    colors = [BEV_COLOR, BEV_COLOR, BEV_COLOR, IMG_COLOR, IMG_COLOR]

    bars = ax.bar(range(len(names)), latencies, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)

    # 100ms budget line
    ax.axhline(y=100, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(4.5, 110, "100 ms\nreal-time\nbudget", fontsize=7, color="red",
            ha="center", va="bottom")

    # 10ms line
    ax.axhline(y=10, color="green", linestyle=":", linewidth=0.8, alpha=0.6)

    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.2,
                f"{val:.1f} ms", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Speedup annotation
    ax.annotate("421× faster", xy=(3, 2.2), xytext=(1.5, 5),
                fontsize=9, fontweight="bold", color=IMG_COLOR,
                arrowprops=dict(arrowstyle="->", color=IMG_COLOR, lw=1.5))

    bev_patch = mpatches.Patch(color=BEV_COLOR, label="BEV-domain")
    img_patch = mpatches.Patch(color=IMG_COLOR, label="Image-space")
    ax.legend(handles=[bev_patch, img_patch], fontsize=8)
    ax.set_title("Planner Latency Comparison (Log Scale)")
    save_thesis_figure(fig, "latency_log_scale", output_dir=FIGURES)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Design Iteration Timeline (NEW)
# ═══════════════════════════════════════════════════════════════════════════

def gen_iteration_timeline():
    fig, ax = plt.subplots(figsize=(8, 3.5))

    versions = ITERATIONS["version"]
    planners = ITERATIONS["planner"]
    domains = ITERATIONS["domain"]
    fps_vals = ITERATIONS["fps"]

    y_pos = range(len(versions))
    colors = [BEV_COLOR, BEV_COLOR, IMG_COLOR, TEMPLATE_COLOR]

    bars = ax.barh(y_pos, fps_vals, color=colors, edgecolor="black", linewidth=0.5, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{v}: {p}\n({d})" for v, p, d in zip(versions, planners, domains)],
                       fontsize=8)
    ax.set_xlabel("Pipeline FPS")
    ax.set_title("Design Iteration Progression")
    ax.axvline(x=10, color="red", linestyle="--", linewidth=1, alpha=0.7, label="10 Hz target")
    ax.legend(fontsize=7)

    for bar, val in zip(bars, fps_vals):
        label = f"{val:.1f} FPS"
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=8, fontweight="bold")

    ax.invert_yaxis()
    save_thesis_figure(fig, "iteration_timeline", output_dir=FIGURES)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8: Full-Video Replay Comparison (NEW — Table 5.2 data)
# ═══════════════════════════════════════════════════════════════════════════

def gen_fullvideo_comparison():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    metrics = FULLVIDEO["metric"]
    baseline = FULLVIDEO["baseline"]
    candidate = FULLVIDEO["candidate"]

    x = np.array([0, 1])
    width = 0.35

    for ax, metric, bval, cval in zip(axes, metrics, baseline, candidate):
        bars_b = ax.bar(x[0], bval, width, color=BASELINE_COLOR, edgecolor="black",
                        linewidth=0.5, label="Baseline")
        bars_c = ax.bar(x[1], cval, width, color=TEMPLATE_COLOR, edgecolor="black",
                        linewidth=0.5, label="Candidate")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "Candidate"], fontsize=8)

        for container, val in [(bars_b, bval), (bars_c, cval)]:
            fmt = f"{val:.3f}" if val < 2 else f"{val:.1f}%"
            rect = container.patches[0]
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.01 * max(abs(bval), abs(cval)),
                    fmt, ha="center", va="bottom", fontsize=7)

    axes[0].legend(fontsize=7)
    fig.suptitle("Full-Video Replay — 22,679 Frames Across 6 Sequences", fontsize=11)
    save_thesis_figure(fig, "fullvideo_comparison", output_dir=FIGURES)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("Generating improved thesis figures...")
    print("=" * 50)

    gen_planner_comparison()
    print("[1/8] planner_comparison_v2.png")

    gen_seg_improvement()
    print("[2/8] seg_improvement_v2.png")

    gen_runtime_breakdown()
    print("[3/8] runtime_breakdown_v2.png")

    gen_training_curves()
    print("[4/8] training_curves_v2.png")

    gen_bev_fragility()
    print("[5/8] bev_fragility_v2.png")

    gen_latency_log_scale()
    print("[6/8] latency_log_scale.png")

    gen_iteration_timeline()
    print("[7/8] iteration_timeline.png")

    gen_fullvideo_comparison()
    print("[8/8] fullvideo_comparison.png")

    print("=" * 50)
    print("Done! All figures saved to thesis/figures/")


if __name__ == "__main__":
    main()
