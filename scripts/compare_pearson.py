#!/usr/bin/env python3
"""
Compute per-recording Pearson r for RGB vs Depth latents across recordings.
Prints a summary table and saves a box plot — no interactive display.

Usage:
    python3 scripts/compare_pearson.py
    python3 scripts/compare_pearson.py --recs 02 03 04 05 06 07 08 09 10 11
    python3 scripts/compare_pearson.py --offset 52 --neighborhood 70
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr


OFFSET       = 0
NEIGHBORHOOD = 60
MAX_L        = 10000


def accm_distances_raw(array, offset, neighborhood, L):
    dist = np.zeros((L + 1) * L // 2)
    for shift in np.arange(1, L // 2):
        window = array[offset : offset + neighborhood]
        rolled = np.roll(window, shift - neighborhood // 2, axis=0)
        dist[(shift - 1) * neighborhood : shift * neighborhood] = \
            np.sqrt(np.sum((window - rolled) ** 2, axis=-1))
    return dist[dist != 0]


def apply_pca(lat: np.ndarray, n_components: int) -> np.ndarray:
    """Per-recording PCA via SVD (no sklearn dependency). Unsupervised --
    fit only on this recording's own extracted latents, no position leakage."""
    centered = lat - lat.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]  # (n_components, D)
    return centered @ components.T  # (N, n_components)


def pearson_for_recording(lat_path, pos_path, offset, neighborhood, pca_dims=None):
    """
    pca_dims: None (default) -- raw flat latent, unchanged behaviour, every
    number produced before this flag existed stays reproducible. If set,
    reduce the latent to its top `pca_dims` principal components (fit per
    recording) before computing distances -- opt-in only, never the default.
    """
    lat = np.load(lat_path).astype(np.float32)
    pos = np.load(pos_path).astype(np.float32)

    if np.any(np.isnan(lat)) or np.any(np.isnan(pos)):
        return None, "NaN in latents/positions"

    if pca_dims is not None:
        if pca_dims >= lat.shape[1]:
            return None, f"pca_dims={pca_dims} >= latent dim {lat.shape[1]}"
        lat = apply_pca(lat, pca_dims)

    L = min(MAX_L, len(lat) - offset)
    if L < neighborhood + 1:
        return None, f"Too few frames ({len(lat)}) for offset={offset}"

    pr = accm_distances_raw(pos, offset, neighborhood, L)
    lr = accm_distances_raw(lat, offset, neighborhood, L)
    n  = min(len(pr), len(lr))
    pr, lr = pr[:n], lr[:n]

    mask = pr > 1e-3
    if mask.sum() < 10:
        return None, "Too few valid pairs"

    r, _ = pearsonr(np.log(pr[mask]), lr[mask])
    return r, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recs', nargs='+', default=[f'{i:02d}' for i in range(2, 12)])
    parser.add_argument('--offset',       type=int, default=OFFSET)
    parser.add_argument('--neighborhood', type=int, default=NEIGHBORHOOD)
    parser.add_argument('--pca', type=int, default=None,
                        help='Reduce latents to top N principal components before scoring (opt-in, off by default)')
    parser.add_argument('--output', type=str, default='data/rgb_vs_depth_pearson.png')
    args = parser.parse_args()

    rgb_pearson   = {}
    depth_pearson = {}

    print(f"\noffset={args.offset}  neighborhood={args.neighborhood}\n")
    print(f"{'Rec':<12} {'RGB':>8} {'Depth':>8}")
    print("-" * 32)

    for rec in args.recs:
        rec_dir = Path(f'data/rec_{rec}')

        # RGB
        rgb_lat  = rec_dir / 'latents_rgb'  / 'latents_flat.npy'
        rgb_pos  = rec_dir / 'latents_rgb'  / 'positions.npy'
        # Depth
        dep_lat  = rec_dir / 'latents'      / 'latents_flat.npy'
        dep_pos  = rec_dir / 'latents'      / 'positions.npy'

        r_rgb = r_dep = None

        if rgb_lat.exists() and rgb_pos.exists():
            r_rgb, err = pearson_for_recording(rgb_lat, rgb_pos, args.offset, args.neighborhood, args.pca)
            if err:
                print(f"  rec_{rec} RGB skipped: {err}")
        else:
            print(f"  rec_{rec} RGB: latents not found")

        if dep_lat.exists() and dep_pos.exists():
            r_dep, err = pearson_for_recording(dep_lat, dep_pos, args.offset, args.neighborhood, args.pca)
            if err:
                print(f"  rec_{rec} Depth skipped: {err}")
        else:
            print(f"  rec_{rec} Depth: latents not found")

        rgb_str = f"{r_rgb:.4f}" if r_rgb is not None else "  N/A"
        dep_str = f"{r_dep:.4f}" if r_dep is not None else "  N/A"
        print(f"rec_{rec:<8} {rgb_str:>8} {dep_str:>8}")

        if r_rgb is not None: rgb_pearson[rec]   = r_rgb
        if r_dep is not None: depth_pearson[rec] = r_dep

    # Summary stats
    print("\n" + "=" * 32)
    if rgb_pearson:
        vals = list(rgb_pearson.values())
        print(f"RGB   — n={len(vals)}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")
    if depth_pearson:
        vals = list(depth_pearson.values())
        print(f"Depth — n={len(vals)}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    # Box plot
    _, ax = plt.subplots(figsize=(6, 5))
    data   = []
    labels = []
    colors = []

    if rgb_pearson:
        data.append(list(rgb_pearson.values()))
        labels.append(f'RGB\n(n={len(rgb_pearson)})')
        colors.append('#4C72B0')

    if depth_pearson:
        data.append(list(depth_pearson.values()))
        labels.append(f'Depth\n(n={len(depth_pearson)})')
        colors.append('#DD8452')

    bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Scatter individual points
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        ax.scatter([i] * len(vals), vals, color=color, zorder=5, s=40, edgecolors='black', linewidths=0.5)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Pearson r (log scale)", fontsize=12)
    ax.set_title("Latent Space Spatial Metric\nRGB vs Depth Predictive Coding", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.827, color='gray', linestyle='--', linewidth=1, label='ACCM (0.827)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\nBox plot saved to {args.output}")

    # Save CSV
    csv_path = Path(args.output).with_suffix('.csv')
    all_recs = sorted(set(list(rgb_pearson.keys()) + list(depth_pearson.keys())))
    with open(csv_path, 'w') as f:
        f.write("recording,rgb_pearson,depth_pearson\n")
        for rec in all_recs:
            r = rgb_pearson.get(rec, '')
            d = depth_pearson.get(rec, '')
            f.write(f"rec_{rec},{r},{d}\n")
    print(f"CSV saved to {csv_path}")


if __name__ == '__main__':
    main()
