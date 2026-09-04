#!/usr/bin/env python3
"""
plot_model_comparison.py
Dual RGB Detection System — Model Comparison Figures

Reads a model_comparison_rgb*.json report (written by
compare_yolo_models_rgb.py) and generates PNG figures:

  1. overall_metrics.png     — mAP50 / mAP50-95 / mask mAP50 per model
  2. per_class_recall.png    — recall per class, grouped by model
  3. per_class_precision.png — precision per class, grouped by model
  4. speed_vs_accuracy.png   — the actual deployment trade-off: inference
                                ms/image (x) vs mAP50-95 (y), one point
                                per model
  5. training_time.png       — wall-clock training time per model
                                (skipped if the report used --skip-training,
                                since there's no training time to show)
  6. confusion_matrices.png  — one heatmap per model, side by side

A model that failed (recorded with an "error" instead of "metrics" in
the report) is shown as a labeled gap in the bar/scatter charts and
skipped in the confusion matrix panel, rather than silently omitted
or crashing the whole figure set.

Usage
-----
    python3 plot_model_comparison.py --report model_comparison_rgb.json

    python3 plot_model_comparison.py \\
        --report model_comparison_rgb_dataset2.json \\
        --out-dir figures_dataset2

    # Compare two datasets' reports side by side (e.g. did the winner
    # change between datasets?)
    python3 plot_model_comparison.py \\
        --report model_comparison_rgb.json model_comparison_rgb_dataset2.json \\
        --labels dataset1 dataset2 \\
        --out-dir figures_combined

Author : Bright Mensah | NDSU / Dual RGB Detection System
Path   : /media/pagsun/Transcend/phd_project/emeet_dual_cam/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless -- no display needed on the Jetson
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


# A small, consistent, colorblind-reasonable palette used across every
# figure so the same model always gets the same color.
_PALETTE = ["#4a9eff", "#00c896", "#f5a623", "#e84545", "#b060d0", "#00a0c0"]


def load_reports(paths, labels):
    """
    Returns a list of (label, report_dict) -- supports plotting one
    report normally, or multiple reports (e.g. two datasets) side by
    side with --labels distinguishing them in legends/titles.
    """
    reports = []
    for i, p in enumerate(paths):
        with open(p) as f:
            data = json.load(f)
        label = labels[i] if labels and i < len(labels) else Path(p).stem
        reports.append((label, data))
    return reports


def _model_entries(report):
    """Yield (model_key, entry) for models that actually have metrics."""
    for key, entry in report.get("results", {}).items():
        yield key, entry


def _color_for(i):
    return _PALETTE[i % len(_PALETTE)]


# ─────────────────────────────────────────────────────────────
#  1. Overall metrics
# ─────────────────────────────────────────────────────────────

def plot_overall_metrics(reports, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    metric_keys = [
        ("mAP50", lambda m: m["overall"]["mAP50"]),
        ("mAP50-95", lambda m: m["overall"]["mAP50_95"]),
        ("mask mAP50", lambda m: (m.get("seg_overall") or {}).get("mask_mAP50")),
    ]

    # Flatten across reports: each (report_label, model_key) is one group.
    groups = []
    for rlabel, report in reports:
        for mkey, entry in _model_entries(report):
            groups.append((rlabel, mkey, entry))

    n_groups = len(groups)
    n_metrics = len(metric_keys)
    x = np.arange(n_groups)
    width = 0.8 / n_metrics

    for mi, (mname, getter) in enumerate(metric_keys):
        values = []
        for _, _, entry in groups:
            if entry.get("error") or not entry.get("metrics"):
                values.append(0.0)
            else:
                v = getter(entry["metrics"])
                values.append(v if v is not None else 0.0)
        ax.bar(x + mi * width - width, values, width, label=mname,
               color=_color_for(mi))

    labels = [f"{mkey}\n({rlabel})" if len(reports) > 1 else mkey
              for rlabel, mkey, _ in groups]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Overall Detection Metrics by Model")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Mark failed models clearly rather than leaving an unexplained zero bar.
    for i, (_, _, entry) in enumerate(groups):
        if entry.get("error"):
            ax.text(i, 0.02, "FAILED", ha="center", va="bottom",
                    rotation=90, fontsize=8, color="#e84545")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  2 & 3. Per-class recall / precision
# ─────────────────────────────────────────────────────────────

def plot_per_class(reports, out_path, metric_field, title):
    fig, ax = plt.subplots(figsize=(11, 6))

    # Class list: take from the first successful model found.
    class_names = None
    for _, report in reports:
        for _, entry in _model_entries(report):
            if entry.get("metrics"):
                class_names = [c["class_name"] for c in
                                entry["metrics"]["per_class"]]
                break
        if class_names:
            break
    if not class_names:
        print(f"  (skipping {out_path.name} -- no successful models with metrics)")
        return

    groups = []
    for rlabel, report in reports:
        for mkey, entry in _model_entries(report):
            groups.append((rlabel, mkey, entry))

    n_groups = len(groups)
    x = np.arange(len(class_names))
    width = 0.8 / max(n_groups, 1)

    for gi, (rlabel, mkey, entry) in enumerate(groups):
        values = []
        if entry.get("error") or not entry.get("metrics"):
            values = [0.0] * len(class_names)
        else:
            by_name = {c["class_name"]: c.get(metric_field, 0.0)
                       for c in entry["metrics"]["per_class"]}
            values = [by_name.get(c, 0.0) for c in class_names]
        label = f"{mkey} ({rlabel})" if len(reports) > 1 else mkey
        ax.bar(x + gi * width - (n_groups - 1) * width / 2, values, width,
               label=label, color=_color_for(gi))

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric_field.capitalize())
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  4. Speed vs accuracy
# ─────────────────────────────────────────────────────────────

def plot_speed_vs_accuracy(reports, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    i = 0
    any_plotted = False
    for rlabel, report in reports:
        for mkey, entry in _model_entries(report):
            if entry.get("error") or not entry.get("metrics"):
                i += 1
                continue
            m = entry["metrics"]
            inf_ms = m.get("speed_ms", {}).get("inference")
            map5095 = m["overall"]["mAP50_95"]
            if inf_ms is None:
                i += 1
                continue
            label = f"{mkey} ({rlabel})" if len(reports) > 1 else mkey
            ax.scatter(inf_ms, map5095, s=140, color=_color_for(i),
                       edgecolors="black", linewidths=0.5, zorder=3)
            ax.annotate(label, (inf_ms, map5095),
                        textcoords="offset points", xytext=(8, 6),
                        fontsize=9)
            any_plotted = True
            i += 1

    if not any_plotted:
        print(f"  (skipping {out_path.name} -- no successful models with "
              f"both speed and accuracy data)")
        return

    ax.set_xlabel("Inference time (ms/image) — lower is better")
    ax.set_ylabel("mAP50-95 — higher is better")
    ax.set_title("Speed vs Accuracy Trade-off")
    ax.grid(alpha=0.3)
    # Arrow hint toward the ideal corner (fast + accurate)
    ax.annotate("", xy=(0.02, 0.98), xycoords="axes fraction",
                xytext=(0.15, 0.85), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#888888"))
    ax.text(0.16, 0.83, "better", transform=ax.transAxes,
            fontsize=8, color="#888888")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  5. Training time
# ─────────────────────────────────────────────────────────────

def plot_training_time(reports, out_path):
    groups = []
    for rlabel, report in reports:
        for mkey, entry in _model_entries(report):
            secs = entry.get("train_seconds")
            if secs is not None:
                groups.append((rlabel, mkey, secs))

    if not groups:
        print(f"  (skipping {out_path.name} -- no training-time data, "
              f"likely a --skip-training report)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{mkey}\n({rlabel})" if len(reports) > 1 else mkey
              for rlabel, mkey, _ in groups]
    hours = [secs / 3600.0 for _, _, secs in groups]
    colors = [_color_for(i) for i in range(len(groups))]

    ax.bar(labels, hours, color=colors)
    ax.set_ylabel("Training time (hours)")
    ax.set_title("Training Time by Model")
    ax.grid(axis="y", alpha=0.3)
    for i, h in enumerate(hours):
        ax.text(i, h + 0.02, f"{h:.1f}h", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  6. Confusion matrices
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrices(reports, out_path):
    panels = []
    for rlabel, report in reports:
        for mkey, entry in _model_entries(report):
            if entry.get("error") or not entry.get("metrics"):
                continue
            cm = entry["metrics"].get("confusion_matrix")
            if not cm:
                continue
            label = f"{mkey} ({rlabel})" if len(reports) > 1 else mkey
            panels.append((label, cm))

    if not panels:
        print(f"  (skipping {out_path.name} -- no confusion matrix data)")
        return

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]

    for ax, (label, cm) in zip(axes, panels):
        matrix = np.array(cm["matrix"])
        row_labels = cm["labels"]
        # Normalize per row (per true class) so the color scale reflects
        # the fraction of each class's instances landing in each bucket
        # -- raw counts would be dominated by whichever class has the
        # most instances and hide the pattern in smaller classes.
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        norm = matrix / row_sums

        im = ax.imshow(norm, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(row_labels)))
        ax.set_yticks(range(len(row_labels)))
        ax.set_xticklabels(row_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(label, fontsize=11)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0:
                    ax.text(j, i, f"{int(matrix[i,j])}", ha="center",
                            va="center", fontsize=7,
                            color="white" if norm[i, j] > 0.5 else "black")

    fig.suptitle("Confusion Matrices (row-normalized; 'background' = missed "
                 "detection or false detection)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", nargs="+", required=True,
                     help="One or more model_comparison_rgb*.json files")
    ap.add_argument("--labels", nargs="+", default=None,
                     help="Labels for each --report (default: filename stem). "
                          "Only meaningful with multiple --report files.")
    ap.add_argument("--out-dir", default=None,
                     help="Output directory for PNG figures. Default: "
                          "a 'comparison_figures' subfolder next to the "
                          "first --report file -- i.e. inside runs/ "
                          "alongside the report it was generated from, "
                          "not the current working directory. Pass an "
                          "explicit path to override.")
    args = ap.parse_args()

    out_dir = (Path(args.out_dir) if args.out_dir is not None
               else Path(args.report[0]).resolve().parent / "comparison_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = load_reports(args.report, args.labels)
    print(f"Loaded {len(reports)} report(s): "
          f"{[l for l, _ in reports]}")

    print("\nGenerating figures...")
    plot_overall_metrics(reports, out_dir / "overall_metrics.png")
    print("  ✓ overall_metrics.png")

    plot_per_class(reports, out_dir / "per_class_recall.png",
                    "recall", "Per-Class Recall by Model")
    print("  ✓ per_class_recall.png")

    plot_per_class(reports, out_dir / "per_class_precision.png",
                    "precision", "Per-Class Precision by Model")
    print("  ✓ per_class_precision.png")

    plot_speed_vs_accuracy(reports, out_dir / "speed_vs_accuracy.png")
    print("  ✓ speed_vs_accuracy.png")

    plot_training_time(reports, out_dir / "training_time.png")
    print("  ✓ training_time.png")

    plot_confusion_matrices(reports, out_dir / "confusion_matrices.png")
    print("  ✓ confusion_matrices.png")

    print(f"\nAll figures written to {out_dir}/")


if __name__ == "__main__":
    main()
