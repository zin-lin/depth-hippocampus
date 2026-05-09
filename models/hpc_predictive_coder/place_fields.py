#!/usr/bin/env python3
"""
Place field analysis for HPCPredictiveCoder latents.

For each latent channel, finds positions where the channel fires in the top 10%
(ACCM methodology), fits a 2D Gaussian, and visualises the place fields.

Usage:
    python3 models/hpc_predictive_coder/place_fields.py \
        --latents data/rec_12/latents/latents_pooled.npy \
        --positions data/rec_12/latents/positions.npy \
        --output data/rec_12/latents/place_fields.png

    # Multiple recordings concatenated
    python3 models/hpc_predictive_coder/place_fields.py \
        --latents data/rec_12/latents/latents_pooled.npy data/rec_13/latents/latents_pooled.npy \
        --positions data/rec_12/latents/positions.npy data/rec_13/latents/positions.npy \
        --output data/place_fields_combined.png
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde


def fit_gaussian_area(positions_2d):
    """Approximate place field area from 2D position covariance (ACCM method)."""
    if len(positions_2d) < 3:
        return 0.0
    try:
        cov = np.cov(positions_2d, rowvar=False)
        if cov.ndim < 2:
            return 0.0
        sv = np.linalg.svd(cov, compute_uv=False)
        return np.multiply.reduce(np.sqrt(np.maximum(sv, 0))) * np.pi
    except Exception:
        return 0.0


def compute_place_fields(latents, positions, quantile=0.9):
    """
    For each channel, find positions where activation > quantile threshold.

    Args:
        latents:   [N, C] pooled latents
        positions: [N, 2] x,y positions
    Returns:
        fields: list of C arrays, each [M, 2] firing positions
        areas:  [C] approximate field areas
    """
    N, C = latents.shape
    fields = []
    areas = []
    for c in range(C):
        thresh = np.quantile(latents[:, c], quantile)
        mask = latents[:, c] > thresh
        firing_pos = positions[mask]
        fields.append(firing_pos)
        areas.append(fit_gaussian_area(firing_pos))
    return fields, np.array(areas)


def plot_fields_grid(fields, positions, output_path, n_cols=16):
    """16×8 grid of place field scatter plots (ACCM paper style)."""
    C = len(fields)
    n_rows = C // n_cols
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.5))
    fig.patch.set_facecolor('#1a1a1a')

    for idx in range(C):
        i, j = idx // n_cols, idx % n_cols
        ax = axes[i, j]
        ax.set_facecolor('#1a1a1a')

        # All trajectory positions (faint)
        ax.scatter(positions[::5, 0], positions[::5, 1],
                   s=0.3, alpha=0.15, color='white')
        # Firing positions
        if len(fields[idx]) > 0:
            ax.scatter(fields[idx][:, 0], fields[idx][:, 1],
                       s=2, alpha=0.6, color='#4fc3f7')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.axis('off')

    plt.suptitle('Place Fields — Latent Units', color='white', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#1a1a1a')
    print(f"Saved field grid to {output_path}")


def plot_coverage_map(fields, positions, output_path):
    """How many units are active at each position."""
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

    # Count active units per trajectory frame
    C = len(fields)
    # rebuild mask: for each unit, which frames fired
    # easier: sum up firing counts per position
    all_firing = np.concatenate(fields, axis=0) if any(len(f) > 0 for f in fields) else np.zeros((0, 2))

    fig, ax = plt.subplots(figsize=(7, 7))
    if len(all_firing) > 0:
        h = ax.hist2d(all_firing[:, 0], all_firing[:, 1],
                      bins=40,
                      range=[[x_min, x_max], [y_min, y_max]],
                      cmap='hot')
        plt.colorbar(h[3], ax=ax, label='Active units count')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Place Unit Coverage Map')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved coverage map to {output_path}")


def plot_area_histogram(areas, output_path):
    """Distribution of place field areas (ACCM paper style)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    valid = areas[areas > 0]
    ax.hist(valid, bins=20, color='#4C72B0', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Field area (m²)')
    ax.set_ylabel('Count')
    ax.set_title(f'Place Field Area Distribution\n'
                 f'median={np.median(valid):.2f}m²  mean={np.mean(valid):.2f}m²')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved area histogram to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latents',   nargs='+', required=True)
    parser.add_argument('--positions', nargs='+', required=True)
    parser.add_argument('--output',    type=str, required=True)
    parser.add_argument('--quantile',  type=float, default=0.9)
    args = parser.parse_args()

    assert len(args.latents) == len(args.positions)

    lat_all, pos_all = [], []
    for lp, pp in zip(args.latents, args.positions):
        l = np.load(lp).astype(np.float32)
        p = np.load(pp).astype(np.float32)
        assert len(l) == len(p)
        lat_all.append(l)
        pos_all.append(p)
        print(f"  {lp}: {l.shape[0]} frames")

    latents   = np.concatenate(lat_all,  axis=0)
    positions = np.concatenate(pos_all, axis=0)
    print(f"\nTotal: {len(latents)} frames, {latents.shape[1]} channels")
    print(f"Environment: x=[{positions[:,0].min():.1f}, {positions[:,0].max():.1f}]m  "
          f"y=[{positions[:,1].min():.1f}, {positions[:,1].max():.1f}]m")

    print(f"\nComputing place fields (top {int(args.quantile*100)}% activations)...")
    fields, areas = compute_place_fields(latents, positions, args.quantile)

    active = sum(1 for f in fields if len(f) > 0)
    print(f"Active units: {active}/128")
    print(f"Field area: median={np.median(areas[areas>0]):.2f}m²  "
          f"mean={np.mean(areas[areas>0]):.2f}m²")

    base = args.output.rsplit('.', 1)
    plot_fields_grid(fields, positions, args.output)
    plot_coverage_map(fields, positions, f"{base[0]}_coverage.{base[1]}")
    plot_area_histogram(areas, f"{base[0]}_areas.{base[1]}")


if __name__ == '__main__':
    main()
