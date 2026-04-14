#!/usr/bin/env python3
"""
ACCM-style spatial metric analysis (Figure 2c).

Uses ACCM's exact sliding window distance computation method.
Expects latents and positions extracted by extract_latents.py.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# -----------------------------
# ACCM sliding window distances
# -----------------------------
def accm_distances_raw(array, offset, neighborhood, L):
    """ACCM's exact fixed-window pairwise distance method.

    Takes a single window array[offset : offset+neighborhood] and computes
    element-wise distances against all circular rolls of that window for
    shifts 1 to L//2.  L controls how many pairs are generated (more L =
    more repeated rolls = more pairs); the window itself never moves.
    offset should skip any stationary warmup at the start of the trajectory.
    """
    dist = np.zeros((L + 1) * L // 2)

    for shift in np.arange(1, L // 2):
        window = array[offset : offset + neighborhood]
        rolled = np.roll(window, shift - neighborhood // 2, axis=0)
        dist[(shift - 1) * neighborhood : shift * neighborhood] = \
            np.sqrt(np.sum((window - rolled) ** 2, axis=-1))

    return dist[dist != 0]


def normalize(array):
    """ACCM's min-max normalization."""
    return (array - array.min()) / (array.max() - array.min())


def main():
    parser = argparse.ArgumentParser(description="ACCM spatial metric analysis")
    parser.add_argument('--latents', type=str, nargs='+', required=True,
                        help='Path(s) to latents_flat.npy — pass multiple to concatenate')
    parser.add_argument('--positions', type=str, nargs='+', required=True,
                        help='Path(s) to positions.npy — must match --latents order')
    parser.add_argument('--offset', type=int, default=0,
                        help='Start offset (0 if no stationary period)')
    parser.add_argument('--neighborhood', type=int, default=90,
                        help='ACCM neighborhood size')
    parser.add_argument('--max_L', type=int, default=10000)
    parser.add_argument('--num_bins', type=int, default=10)
    parser.add_argument('--output', type=str, default='spatial_analysis.png')
    parser.add_argument('--raw', action='store_true',
                        help='Use raw L2 distances (no normalization)')
    args = parser.parse_args()

    if len(args.latents) != len(args.positions):
        raise ValueError("Number of --latents and --positions paths must match")

    # Load data per trajectory — distances computed per-trajectory to avoid
    # cross-trajectory boundary contamination when concatenating
    print("Loading data...")
    lat_arrays, pos_arrays = [], []
    for lat_path, pos_path in zip(args.latents, args.positions):
        l = np.load(lat_path).astype(np.float32)
        p = np.load(pos_path).astype(np.float32)
        assert len(l) == len(p), f"Mismatch in {lat_path}: {len(l)} latents vs {len(p)} positions"
        lat_arrays.append(l)
        pos_arrays.append(p)
        print(f"  {lat_path}: {l.shape[0]} frames")

    # Compute distances independently per trajectory, then pool pairs
    print(f"\noffset={args.offset}, neighborhood={args.neighborhood}")
    all_pos_raw, all_lat_raw = [], []
    for i, (lat, pos) in enumerate(zip(lat_arrays, pos_arrays)):
        L = min(args.max_L, len(lat) - args.offset)
        print(f"\n  [{i+1}/{len(lat_arrays)}] L={L} frames")
        print(f"  Computing physical distances...")
        pr = accm_distances_raw(pos, args.offset, args.neighborhood, L)
        print(f"  Computing latent distances...")
        lr = accm_distances_raw(lat, args.offset, args.neighborhood, L)
        min_len = min(len(pr), len(lr))
        all_pos_raw.append(pr[:min_len])
        all_lat_raw.append(lr[:min_len])

    pos_raw = np.concatenate(all_pos_raw)
    lat_raw = np.concatenate(all_lat_raw)

    position_dists = pos_raw

    if args.raw:
        latent_dists = lat_raw
        ylabel = "Latent distance (ℓ₂)"
    else:
        latent_dists = normalize(lat_raw)
        ylabel = "Latent distance (a.u.)"

    # Sort by position distance
    idx = np.argsort(position_dists)
    position_dists = position_dists[idx]
    latent_dists = latent_dists[idx]

    print(f"Computed {len(position_dists)} distance pairs")
    print(f"Physical dist range: [{position_dists.min():.4f}, {position_dists.max():.4f}] m")
    print(f"Latent dist range:   [{latent_dists.min():.6f}, {latent_dists.max():.6f}]")

    # Bin statistics
    edges = np.quantile(position_dists, np.linspace(0, 1, args.num_bins + 1))
    edges = np.unique(edges)
    digitized = np.digitize(position_dists, edges, right=False)

    bin_centers, medians, q25, q75 = [], [], [], []
    for b in range(1, len(edges)):
        vals = latent_dists[digitized == b]
        if vals.size > 30:
            bin_centers.append(0.5 * (edges[b - 1] + edges[b]))
            medians.append(np.median(vals))
            q25.append(np.percentile(vals, 25))
            q75.append(np.percentile(vals, 75))

    # ACCM weighted log regression
    mask = position_dists > 1e-3

    def weighted_log_loss(params):
        alpha, beta = params
        pred = alpha * np.log(position_dists[mask]) + beta
        weight = 1 / (position_dists[mask] + 1e-6)
        return np.mean(weight * (pred - latent_dists[mask]) ** 2)

    result = minimize(weighted_log_loss, [1.0, 0.0])
    alpha, beta = result.x
    print(f"\nRegression: {alpha:.3f} * log(d) + {beta:.3f}")

    # Correlation
    from scipy.stats import pearsonr
    r, p = pearsonr(position_dists[mask], latent_dists[mask])
    print(f"Pearson r = {r:.4f}, p = {p:.2e}")

    d_fit = np.linspace(max(position_dists.min(), 1e-3), position_dists.max(), 300)
    l_fit = alpha * np.log(d_fit) + beta

    # Figure 1: Linear x-axis (ACCM paper style)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.scatter(position_dists[::10], latent_dists[::10],
                s=15, alpha=0.15, color='steelblue', label='Predictive coding')
    ax1.plot(d_fit, l_fit, "k--", linewidth=2, label='log fit')
    ax1.set_xlabel("Actual distance (m)", fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title("Latent space recovers environment's metric", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    output_base = args.output.rsplit('.', 1)
    fig1.savefig(f"{output_base[0]}_linear.{output_base[1]}", dpi=300, bbox_inches='tight')
    print(f"\nSaved linear plot to {output_base[0]}_linear.{output_base[1]}")

    # Figure 2: Log x-axis — filter out small distances
    log_mask = position_dists > 1e-2
    pos_log = position_dists[log_mask]
    lat_log = latent_dists[log_mask]

    # Refit regression on cropped data
    def weighted_log_loss_crop(params):
        a, b = params
        pred = a * np.log(pos_log) + b
        weight = 1 / (pos_log + 1e-6)
        return np.mean(weight * (pred - lat_log) ** 2)

    result_crop = minimize(weighted_log_loss_crop, [1.0, 0.0])
    a2, b2 = result_crop.x

    r2, _ = pearsonr(pos_log, lat_log)
    print(f"Log plot regression: {a2:.3f} * log(d) + {b2:.3f}")
    print(f"Log plot Pearson r = {r2:.4f}")

    d_fit2 = np.linspace(pos_log.min(), pos_log.max(), 300)
    l_fit2 = a2 * np.log(d_fit2) + b2

    lat_clip = np.percentile(lat_log, 99)
    clip_mask = lat_log <= lat_clip
    pos_plot = pos_log[clip_mask]
    lat_plot = lat_log[clip_mask]

    fig2, ax2 = plt.subplots(figsize=(6, 15))
    ax2.scatter(pos_plot[::10], lat_plot[::10],
                s=26, alpha=0.2, color='steelblue', label='Predictive coding')
    ax2.plot(d_fit2, l_fit2, "k--", linewidth=2, label='log fit')
    ax2.set_xscale("log")
    ax2.set_xlim(pos_log.min() * 0.9, pos_log.max() * 1.2)
    ax2.set_ylim(bottom=0.00)
    ax2.set_xlabel("Actual distance (m)", fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.set_title(f"Log scale (r={r2:.3f})", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{output_base[0]}_log.{output_base[1]}", dpi=300, bbox_inches='tight')
    print(f"Saved log plot to {output_base[0]}_log.{output_base[1]}")

    plt.show()


if __name__ == "__main__":
    main()
