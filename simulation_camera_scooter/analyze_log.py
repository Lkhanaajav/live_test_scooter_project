#!/usr/bin/env python3
"""
analyze_log.py
==============
Reads CSV logs from live_heading_demo.py and generates:
  1. Summary statistics (printed + saved as LaTeX table)
  2. Per-module timing breakdown plot
  3. Heading angle over time plot
  4. Speed over time plot
  5. FPS over time plot
  6. Detection frequency histogram
  7. Command distribution pie chart
  8. Ablation comparison (if multiple logs provided)

Usage:
    python analyze_log.py logs/run_20260209_143000.csv
    python analyze_log.py logs/run_*.csv                   # compare multiple runs
    python analyze_log.py logs/run_*.csv --output figures/  # save to thesis figures
"""

import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd

# ---------- Optional: matplotlib ----------
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for saving
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")
    print("         Text statistics will still be generated.\n")


def load_log(csv_path):
    """Load a single CSV log into a DataFrame."""
    df = pd.read_csv(csv_path)
    # Convert empty strings to NaN for numeric columns
    numeric_cols = [
        "t_segmentation", "t_detection", "t_bev", "t_skeleton",
        "t_pathfinding", "t_gps_fusion", "t_command", "t_total_pipeline",
        "fps", "heading_raw_deg", "heading_smoothed_deg",
        "speed_raw_mps", "speed_smoothed_mps",
        "has_path", "num_paths", "best_path_length_px",
        "num_graph_nodes", "num_graph_edges",
        "num_detections", "min_obstacle_dist_m",
        "gps_lat", "gps_lon", "gps_fix_quality",
        "gps_wp_dist_m", "gps_correction_deg",
        "sidewalk_mask_pixels", "bev_mask_pixels", "skeleton_pixels",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# =============================================================================
# 1. Summary statistics
# =============================================================================
def print_summary(df, label=""):
    """Print thesis-ready summary statistics."""
    n = len(df)
    header = f"\n{'='*60}\n  SUMMARY: {label} ({n} frames)\n{'='*60}"
    print(header)

    # --- Timing ---
    timing_cols = {
        "Segmentation": "t_segmentation",
        "Detection": "t_detection",
        "BEV Projection": "t_bev",
        "Skeleton+Graph": "t_skeleton",
        "Path Finding": "t_pathfinding",
        "GPS Fusion": "t_gps_fusion",
        "Command+Smooth": "t_command",
        "TOTAL Pipeline": "t_total_pipeline",
    }
    print("\n  Per-Module Timing (ms):")
    print(f"  {'Module':<20s} {'Mean':>8s} {'Std':>8s} {'Med':>8s} {'P95':>8s} {'Max':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, col in timing_cols.items():
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                print(f"  {name:<20s} {s.mean():8.1f} {s.std():8.1f} "
                      f"{s.median():8.1f} {s.quantile(0.95):8.1f} {s.max():8.1f}")

    # --- FPS ---
    fps = df["fps"].dropna()
    if len(fps) > 0:
        print(f"\n  FPS: mean={fps.mean():.1f}, std={fps.std():.1f}, "
              f"min={fps.min():.1f}, max={fps.max():.1f}")

    # --- Heading ---
    heading = df["heading_smoothed_deg"].dropna()
    if len(heading) > 0:
        print(f"\n  Heading (smoothed): mean={heading.mean():.1f} deg, "
              f"std={heading.std():.1f}, range=[{heading.min():.1f}, {heading.max():.1f}]")

    # --- Speed ---
    speed = df["speed_smoothed_mps"].dropna()
    if len(speed) > 0:
        print(f"  Speed (smoothed): mean={speed.mean():.2f} m/s, "
              f"std={speed.std():.2f}, range=[{speed.min():.2f}, {speed.max():.2f}]")

    # --- Command distribution ---
    if "command" in df.columns:
        cmds = df["command"].value_counts()
        total = cmds.sum()
        print(f"\n  Command Distribution:")
        for cmd, count in cmds.items():
            print(f"    {cmd:<15s}: {count:5d} ({100*count/total:5.1f}%)")

    # --- Path availability ---
    has_path = df["has_path"].dropna()
    if len(has_path) > 0:
        pct = has_path.mean() * 100
        print(f"\n  Path availability: {pct:.1f}% of frames had a valid path")

    # --- Detections ---
    det = df["num_detections"].dropna()
    if len(det) > 0:
        frames_with_det = (det > 0).sum()
        print(f"  Detection: {frames_with_det}/{n} frames had >= 1 object "
              f"({100*frames_with_det/n:.1f}%)")
        if frames_with_det > 0:
            obs_dist = df["min_obstacle_dist_m"].dropna()
            if len(obs_dist) > 0:
                print(f"  Closest obstacle: mean={obs_dist.mean():.1f}m, "
                      f"min={obs_dist.min():.1f}m")

    # --- Mask stats ---
    sw = df["sidewalk_mask_pixels"].dropna()
    if len(sw) > 0:
        print(f"\n  Sidewalk mask: mean={sw.mean():.0f} px, std={sw.std():.0f}")
    bev = df["bev_mask_pixels"].dropna()
    if len(bev) > 0:
        print(f"  BEV mask:      mean={bev.mean():.0f} px, std={bev.std():.0f}")

    print()
    return timing_cols


# =============================================================================
# 2. Generate LaTeX table
# =============================================================================
def generate_latex_table(df, output_path, label=""):
    """Generate a LaTeX table of per-module timing for the thesis."""
    timing_cols = {
        "SegFormer Inference": "t_segmentation",
        "YOLOv8-nano Detection": "t_detection",
        "BEV Projection + Cleaning": "t_bev",
        "Skeleton + Graph": "t_skeleton",
        "Path Finding": "t_pathfinding",
        "GPS Fusion": "t_gps_fusion",
        "Command + Smoothing": "t_command",
    }

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-module runtime breakdown" +
                 (f" ({label})" if label else "") + r".}")
    lines.append(r"\label{tab:runtime_breakdown}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Module & Mean (ms) & Std (ms) & P95 (ms) & \% of Total \\")
    lines.append(r"\midrule")

    total_mean = df["t_total_pipeline"].dropna().mean()

    for name, col in timing_cols.items():
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0 and s.mean() > 0.01:
                pct = 100.0 * s.mean() / total_mean if total_mean > 0 else 0
                lines.append(f"{name} & {s.mean():.1f} & {s.std():.1f} & "
                             f"{s.quantile(0.95):.1f} & {pct:.1f}\\% \\\\")

    lines.append(r"\midrule")
    lines.append(f"Total Pipeline & {total_mean:.1f} & "
                 f"{df['t_total_pipeline'].dropna().std():.1f} & "
                 f"{df['t_total_pipeline'].dropna().quantile(0.95):.1f} & 100.0\\% \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(latex_str)
    print(f"  LaTeX table saved to {output_path}")
    return latex_str


# =============================================================================
# 3-7. Plots
# =============================================================================
def plot_timing_breakdown(df, output_dir, label=""):
    """Bar chart: per-module mean timing."""
    if not HAS_MPL:
        return
    modules = {
        "Segmentation": "t_segmentation",
        "Detection": "t_detection",
        "BEV": "t_bev",
        "Skeleton": "t_skeleton",
        "Pathfinding": "t_pathfinding",
        "GPS": "t_gps_fusion",
        "Command": "t_command",
    }
    names, means, stds = [], [], []
    for name, col in modules.items():
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0 and s.mean() > 0.01:
                names.append(name)
                means.append(s.mean())
                stds.append(s.std())

    if not names:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.barh(names, means, xerr=stds, color=colors, edgecolor="gray",
                   capsize=3, height=0.6)
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"Per-Module Timing Breakdown" + (f" — {label}" if label else ""))
    ax.invert_yaxis()
    for bar, val in zip(bars, means):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "timing_breakdown.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


def plot_heading_over_time(df, output_dir, label=""):
    """Line plot: heading angle over frames."""
    if not HAS_MPL:
        return
    sub = df[["frame_id", "heading_raw_deg", "heading_smoothed_deg"]].dropna()
    if len(sub) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(sub["frame_id"], sub["heading_raw_deg"], alpha=0.3, linewidth=0.8,
            label="Raw", color="gray")
    ax.plot(sub["frame_id"], sub["heading_smoothed_deg"], linewidth=1.2,
            label="Smoothed (K=5)", color="royalblue")
    ax.axhline(0, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(12, color="orange", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.axhline(-12, color="orange", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.axhline(40, color="red", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.axhline(-40, color="red", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Heading (deg)")
    ax.set_title(f"Heading Angle Over Time" + (f" — {label}" if label else ""))
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-90, 90)
    plt.tight_layout()
    path = os.path.join(output_dir, "heading_over_time.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


def plot_speed_over_time(df, output_dir, label=""):
    """Line plot: speed over frames."""
    if not HAS_MPL:
        return
    sub = df[["frame_id", "speed_smoothed_mps"]].dropna()
    if len(sub) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(sub["frame_id"], 0, sub["speed_smoothed_mps"],
                    alpha=0.3, color="steelblue")
    ax.plot(sub["frame_id"], sub["speed_smoothed_mps"], linewidth=1,
            color="steelblue", label="Speed (m/s)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title(f"Speed Over Time" + (f" — {label}" if label else ""))
    ax.set_ylim(0, 2.0)
    plt.tight_layout()
    path = os.path.join(output_dir, "speed_over_time.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


def plot_fps_over_time(df, output_dir, label=""):
    """Line plot: FPS stability."""
    if not HAS_MPL:
        return
    sub = df[["frame_id", "fps"]].dropna()
    if len(sub) == 0:
        return
    fps_mean = sub["fps"].mean()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(sub["frame_id"], sub["fps"], linewidth=0.8, color="seagreen")
    ax.axhline(fps_mean, color="red", linestyle="--", linewidth=1,
               label=f"Mean: {fps_mean:.1f} FPS")
    ax.set_xlabel("Frame")
    ax.set_ylabel("FPS")
    ax.set_title(f"Pipeline FPS Over Time" + (f" — {label}" if label else ""))
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "fps_over_time.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


def plot_command_distribution(df, output_dir, label=""):
    """Pie chart: command distribution."""
    if not HAS_MPL:
        return
    cmds = df["command"].value_counts()
    if len(cmds) == 0:
        return
    colors_map = {
        "STRAIGHT": "#2ecc71", "LEFT": "#f39c12", "RIGHT": "#3498db",
        "SHARP LEFT": "#e74c3c", "SHARP RIGHT": "#c0392b", "STOP": "#7f8c8d",
    }
    colors = [colors_map.get(c, "#bdc3c7") for c in cmds.index]

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        cmds.values, labels=cmds.index, autopct="%1.1f%%",
        colors=colors, startangle=90, textprops={"fontsize": 9})
    ax.set_title(f"Command Distribution" + (f"\n{label}" if label else ""))
    plt.tight_layout()
    path = os.path.join(output_dir, "command_distribution.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


def plot_detection_histogram(df, output_dir, label=""):
    """Histogram: obstacle distance distribution."""
    if not HAS_MPL:
        return
    dists = df["min_obstacle_dist_m"].dropna()
    if len(dists) == 0:
        print("  (no obstacle detections to plot)")
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(dists, bins=30, color="coral", edgecolor="white", alpha=0.8)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="Stop zone (1m)")
    ax.axvline(3.0, color="orange", linestyle="--", linewidth=1.5, label="Slow zone (3m)")
    ax.set_xlabel("Closest Obstacle Distance (m)")
    ax.set_ylabel("Frame Count")
    ax.set_title(f"Obstacle Distance Distribution" + (f" — {label}" if label else ""))
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "obstacle_distances.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# 8. Ablation comparison (multiple logs)
# =============================================================================
def plot_ablation_comparison(dfs, labels, output_dir):
    """Compare FPS and heading std across multiple runs (ablation study)."""
    if not HAS_MPL or len(dfs) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # FPS comparison
    fps_means = [df["fps"].dropna().mean() for df in dfs]
    fps_stds = [df["fps"].dropna().std() for df in dfs]
    axes[0].bar(labels, fps_means, yerr=fps_stds, color=plt.cm.Set2(np.linspace(0, 1, len(dfs))),
                edgecolor="gray", capsize=4)
    axes[0].set_ylabel("FPS")
    axes[0].set_title("Pipeline Speed")

    # Heading std comparison (smoothness)
    h_stds = [df["heading_smoothed_deg"].dropna().std() for df in dfs]
    axes[1].bar(labels, h_stds, color=plt.cm.Set2(np.linspace(0, 1, len(dfs))),
                edgecolor="gray")
    axes[1].set_ylabel("Heading Std (deg)")
    axes[1].set_title("Heading Stability (lower=better)")

    # Path availability comparison
    path_pcts = [df["has_path"].dropna().mean() * 100 for df in dfs]
    axes[2].bar(labels, path_pcts, color=plt.cm.Set2(np.linspace(0, 1, len(dfs))),
                edgecolor="gray")
    axes[2].set_ylabel("Path Available (%)")
    axes[2].set_title("Path Availability")
    axes[2].set_ylim(0, 105)

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Ablation Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment logs from live_heading_demo.py")
    parser.add_argument("csv_files", nargs="+",
                        help="One or more CSV log files (supports glob)")
    parser.add_argument("--output", "-o", type=str, default="figures",
                        help="Output directory for plots and tables (default: figures/)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Only print text statistics, skip plots")
    args = parser.parse_args()

    # Expand globs (Windows doesn't auto-expand)
    csv_files = []
    for pattern in args.csv_files:
        expanded = glob.glob(pattern)
        if expanded:
            csv_files.extend(expanded)
        else:
            csv_files.append(pattern)

    if not csv_files:
        print("No CSV files found.")
        return

    os.makedirs(args.output, exist_ok=True)

    dfs = []
    labels = []
    for csv_path in sorted(csv_files):
        if not os.path.isfile(csv_path):
            print(f"WARNING: {csv_path} not found, skipping.")
            continue
        print(f"\nLoading: {csv_path}")
        df = load_log(csv_path)
        label = os.path.splitext(os.path.basename(csv_path))[0]
        dfs.append(df)
        labels.append(label)

        # Print summary
        print_summary(df, label=label)

        # Generate LaTeX table
        latex_path = os.path.join(args.output, f"{label}_table.tex")
        generate_latex_table(df, latex_path, label=label)

        # Generate plots
        if not args.no_plots and HAS_MPL:
            print(f"\n  Generating plots for {label}...")
            run_output = os.path.join(args.output, label)
            os.makedirs(run_output, exist_ok=True)
            plot_timing_breakdown(df, run_output, label)
            plot_heading_over_time(df, run_output, label)
            plot_speed_over_time(df, run_output, label)
            plot_fps_over_time(df, run_output, label)
            plot_command_distribution(df, run_output, label)
            plot_detection_histogram(df, run_output, label)

    # Ablation comparison if multiple runs
    if len(dfs) >= 2 and not args.no_plots and HAS_MPL:
        print("\n  Generating ablation comparison...")
        plot_ablation_comparison(dfs, labels, args.output)

    print(f"\nAll done. Output in: {os.path.abspath(args.output)}/")


if __name__ == "__main__":
    main()
