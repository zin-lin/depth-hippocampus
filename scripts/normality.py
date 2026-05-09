#!/usr/bin/env python3
"""
Normality test for RGB vs Depth Pearson distributions.
Reads CSV from compare_pearson.py and runs Shapiro-Wilk on each group.
Recommends T-test or Mann-Whitney U accordingly.

Usage:
    python3 scripts/normality.py
    python3 scripts/normality.py --csv data/rgb_vs_depth_pearson.csv
"""

import argparse
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import shapiro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/rgb_vs_depth_pearson.csv')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run compare_pearson.py first.")
        return

    rgb_vals, depth_vals = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['rgb_pearson']:
                rgb_vals.append(float(row['rgb_pearson']))
            if row['depth_pearson']:
                depth_vals.append(float(row['depth_pearson']))

    print(f"\nLoaded {len(rgb_vals)} RGB and {len(depth_vals)} Depth Pearson values from {csv_path}\n")
    print("=" * 50)
    print("Shapiro-Wilk Normality Test")
    print("=" * 50)

    rgb_normal = depth_normal = False

    if len(rgb_vals) >= 3:
        stat, p = shapiro(rgb_vals)
        rgb_normal = p > 0.05
        print(f"\nRGB:")
        print(f"  values : {[round(v, 4) for v in rgb_vals]}")
        print(f"  W={stat:.4f}, p={p:.4f}")
        print(f"  → {'NORMAL (p > 0.05)' if rgb_normal else 'NOT normal (p <= 0.05)'}")
    else:
        print(f"\nRGB: too few values (n={len(rgb_vals)})")

    if len(depth_vals) >= 3:
        stat, p = shapiro(depth_vals)
        depth_normal = p > 0.05
        print(f"\nDepth:")
        print(f"  values : {[round(v, 4) for v in depth_vals]}")
        print(f"  W={stat:.4f}, p={p:.4f}")
        print(f"  → {'NORMAL (p > 0.05)' if depth_normal else 'NOT normal (p <= 0.05)'}")
    else:
        print(f"\nDepth: too few values (n={len(depth_vals)})")

    print("\n" + "=" * 50)
    if rgb_normal and depth_normal:
        from scipy.stats import ttest_ind
        print("Both distributions normal → Independent T-test")
        stat, p = ttest_ind(rgb_vals, depth_vals)
        print(f"\n  t={stat:.4f}, p={p:.4f}")
    else:
        from scipy.stats import mannwhitneyu
        print("At least one not normal → Mann-Whitney U test")
        stat, p = mannwhitneyu(rgb_vals, depth_vals, alternative='two-sided')
        print(f"\n  U={stat:.4f}, p={p:.4f}")

    sig = p < 0.05
    print(f"  → {'Significant difference (p < 0.05)' if sig else 'No significant difference (p >= 0.05)'}")
    print("=" * 50)

    # Overlapping KDE distribution plot (Mann-Whitney U style)
    from scipy.stats import gaussian_kde

    csv_path_obj = Path(args.csv)
    out_path = csv_path_obj.with_name('significance_test.png')

    _, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(0, 1.05, 500)
    for vals, color, label in [
        (rgb_vals,   '#4C72B0', f'RGB (n={len(rgb_vals)})'),
        (depth_vals, '#DD8452', f'Depth (n={len(depth_vals)})'),
    ]:
        kde = gaussian_kde(vals, bw_method=0.4)
        y   = kde(x)
        ax.fill_between(x, y, alpha=0.4, color=color)
        ax.plot(x, y, color=color, linewidth=2, label=label)
        # Rug plot — individual data points
        ax.plot(vals, [-0.05] * len(vals), '|', color=color, markersize=14, markeredgewidth=2)

    test_name = "Mann-Whitney U" if not (rgb_normal and depth_normal) else "Independent T-test"
    p_str = f"p={p:.4f}{'  *' if sig else '  ns'}"
    ax.set_xlabel("Pearson r", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"RGB vs Depth — {test_name}  |  {p_str}", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nDistribution plot saved to {out_path}")


if __name__ == '__main__':
    main()
