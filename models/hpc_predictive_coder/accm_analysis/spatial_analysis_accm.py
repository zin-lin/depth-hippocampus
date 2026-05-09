#!/usr/bin/env python3
"""
Spatial analysis on ACCM val-set latents using ACCM's exact methodology.

Replicates analysis.py Latents class defaults:
    offset=100, neighborhood=90, L=10000
    euclidean distance on flattened latents [N, 8192]
    normalized latent distances (min-max)
    log-Pearson r (matches ACCM's reported r=0.827)
    MI: 2D histogram 10 bins, second-half of pairs

Usage:
    python3 models/hpc_predictive_coder/accm_analysis/spatial_analysis_accm.py \
        --latents data/accm/latents/latents_flat.npy \
        --positions data/accm/latents/positions.npy
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import minimize
from pathlib import Path


def accm_distances_raw(array, offset, neighborhood, L):
    """ACCM's exact fixed-window pairwise distance method (from analysis.py _dist)."""
    dist = np.zeros((L + 1) * L // 2)
    for shift in np.arange(1, L // 2):
        window = array[offset : offset + neighborhood]
        rolled = np.roll(window, shift - neighborhood // 2, axis=0)
        dist[(shift - 1) * neighborhood : shift * neighborhood] = \
            np.sqrt(np.sum((window - rolled) ** 2, axis=-1))
    return dist[dist != 0]


def normalize(array):
    return (array - array.min()) / (array.max() - array.min())


def compute_mi(pos_dists, lat_dists, bins=10):
    """ACCM's MI: second half of pairs, 10-bin 2D histogram."""
    n = len(pos_dists)
    p_xy = np.histogram2d(pos_dists[n // 2:], lat_dists[n // 2:], bins=bins)[0]
    p_xy /= p_xy.sum()
    p_x = p_xy.sum(axis=0)
    p_y = p_xy.sum(axis=1)
    marginal = p_y[:, None] @ p_x[None, :]
    p_idx = (p_xy != 0) & (marginal != 0)
    return float(np.sum((p_xy * (np.log2(p_xy) - np.log2(marginal)))[p_idx]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latents',      type=str, required=True,
                        help='Path to latents_flat.npy [N, 8192]')
    parser.add_argument('--positions',    type=str, required=True,
                        help='Path to positions.npy [N, 2]')
    parser.add_argument('--offset',       type=int, default=100)
    parser.add_argument('--neighborhood', type=int, default=90)
    parser.add_argument('--L',            type=int, default=10000)
    parser.add_argument('--output',       type=str, default='accm_spatial_analysis.png')
    args = parser.parse_args()

    latents   = np.load(args.latents).astype(np.float32)    # [N, 8192]
    positions = np.load(args.positions).astype(np.float32)  # [N, 2]
    N = len(latents)
    L = min(args.L, N - args.offset)

    print(f"Loaded: {N} episodes")
    print(f"offset={args.offset}  neighborhood={args.neighborhood}  L={L}")

    print("Computing position distances...")
    pos_raw = accm_distances_raw(positions, args.offset, args.neighborhood, L)

    print("Computing latent distances...")
    lat_raw = accm_distances_raw(latents,   args.offset, args.neighborhood, L)

    n = min(len(pos_raw), len(lat_raw))
    pos_raw, lat_raw = pos_raw[:n], lat_raw[:n]

    lat_norm = normalize(lat_raw)

    # Sort by position distance (ACCM's distances property does this)
    idx = np.argsort(pos_raw)
    pos_sorted = pos_raw[idx]
    lat_sorted = lat_norm[idx]

    # Log-Pearson r (ACCM's reported metric)
    mask = pos_sorted > 1e-3
    r_log, p_log = pearsonr(np.log(pos_sorted[mask]), lat_sorted[mask])
    print(f"\nLog-Pearson r = {r_log:.4f}  (p={p_log:.2e})  [ACCM reports 0.827]")

    # MI
    mi = compute_mi(pos_sorted, lat_sorted)
    print(f"MI            = {mi:.4f} bits         [ACCM reports 0.627]")

    # Weighted log regression
    def weighted_log_loss(params):
        a, b = params
        pred = a * np.log(pos_sorted[mask]) + b
        w = 1 / (pos_sorted[mask] + 1e-6)
        return np.mean(w * (pred - lat_sorted[mask]) ** 2)

    res = minimize(weighted_log_loss, [1.0, 0.0])
    alpha, beta = res.x
    print(f"Regression    = {alpha:.3f} * log(d) + {beta:.3f}")

    # Plot
    d_fit = np.linspace(max(pos_sorted.min(), 1e-3), pos_sorted.max(), 300)
    l_fit = alpha * np.log(d_fit) + beta

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    ax = axes[0]
    ax.scatter(pos_sorted[::10], lat_sorted[::10], s=8, alpha=0.15, color='steelblue')
    ax.plot(d_fit, l_fit, 'k--', lw=2, label='log fit')
    ax.set_xlabel("Physical distance (lattice units)")
    ax.set_ylabel("Latent distance (a.u.)")
    ax.set_title(f"ACCM val set — our model  (r={r_log:.3f}, MI={mi:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale
    ax = axes[1]
    log_mask = pos_sorted > 1e-2
    ax.scatter(pos_sorted[log_mask][::10], lat_sorted[log_mask][::10],
               s=8, alpha=0.15, color='steelblue')
    ax.plot(d_fit, l_fit, 'k--', lw=2)
    ax.set_xscale('log')
    ax.set_xlabel("Physical distance (lattice units, log scale)")
    ax.set_ylabel("Latent distance (a.u.)")
    ax.set_title(f"Log scale  (offset={args.offset}, n={args.neighborhood})")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {args.output}")
    plt.show()


if __name__ == '__main__':
    main()
