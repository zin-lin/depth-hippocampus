#!/usr/bin/env python3
"""
Compare Pearson r values from two CSV files.
Usage:
    python3 scripts/compare_csvs.py data/a.csv data/b.csv
    python3 scripts/compare_csvs.py data/a.csv data/b.csv --col depth_pearson --label_a "stride=1" --label_b "stride=10"
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from scipy.stats import shapiro, ttest_ind, mannwhitneyu


def load_col(path, col):
    vals = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get(col):
                try:
                    vals.append(float(row[col]))
                except ValueError:
                    pass
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_a')
    parser.add_argument('csv_b')
    parser.add_argument('--col',     default='depth_pearson')
    parser.add_argument('--label_a', default=None)
    parser.add_argument('--label_b', default=None)
    args = parser.parse_args()

    label_a = args.label_a or Path(args.csv_a).stem
    label_b = args.label_b or Path(args.csv_b).stem

    a = load_col(args.csv_a, args.col)
    b = load_col(args.csv_b, args.col)

    print(f"\n{label_a} (n={len(a)}): {[round(v,4) for v in a]}")
    print(f"  mean={np.mean(a):.4f}  std={np.std(a):.4f}")
    print(f"\n{label_b} (n={len(b)}): {[round(v,4) for v in b]}")
    print(f"  mean={np.mean(b):.4f}  std={np.std(b):.4f}")

    print("\nShapiro-Wilk:")
    na = nb = False
    if len(a) >= 3:
        s, p = shapiro(a)
        na = p > 0.05
        print(f"  {label_a}: W={s:.4f}, p={p:.4f} → {'normal' if na else 'NOT normal'}")
    if len(b) >= 3:
        s, p = shapiro(b)
        nb = p > 0.05
        print(f"  {label_b}: W={s:.4f}, p={p:.4f} → {'normal' if nb else 'NOT normal'}")

    print()
    if na and nb:
        stat, p = ttest_ind(a, b)
        print(f"Independent T-test: t={stat:.4f}, p={p:.4f}", end="")
    else:
        stat, p = mannwhitneyu(a, b, alternative='two-sided')
        print(f"Mann-Whitney U: U={stat:.4f}, p={p:.4f}", end="")

    print(f"  → {'Significant *' if p < 0.05 else 'Not significant'}")


if __name__ == '__main__':
    main()
