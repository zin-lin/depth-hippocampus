#!/usr/bin/env python3
"""
Compute mutual information between latent distances and physical distances.
Matches ACCM's methodology (2D histogram, 10 bins).

Columns (--mode all):  Depth PC | Proposed Mech | RGB PC
Proposed Mech: binarised pooled latents (which units fire), ACCM's Figure 2e baseline.

Usage:
    python3 scripts/mutual_information.py --rec 12
    python3 scripts/mutual_information.py --rec 12 --mode rgb
    python3 scripts/mutual_information.py --recs 12 13 14 --mode all
    python3 scripts/mutual_information.py --recs 12 13 14 15 16 17 18 19 20 21
"""

import argparse
import numpy as np
from pathlib import Path


def accm_distances_raw(array, offset, neighborhood, L):
    dist = np.zeros((L + 1) * L // 2)
    for shift in np.arange(1, L // 2):
        window = array[offset : offset + neighborhood]
        rolled = np.roll(window, shift - neighborhood // 2, axis=0)
        dist[(shift - 1) * neighborhood : shift * neighborhood] = \
            np.sqrt(np.sum((window - rolled) ** 2, axis=-1))
    return dist[dist != 0]


def compute_mi(latents, positions, offset=0, neighborhood=50, L=1200, shuffle=False):
    N = len(positions)
    L = min(L, N)

    if shuffle:
        idx = np.random.permutation(N)
        latents, positions = latents[idx], positions[idx]

    pos_dists = accm_distances_raw(positions, offset, neighborhood, L)
    lat_dists = accm_distances_raw(latents, offset, neighborhood, L)
    n = min(len(pos_dists), len(lat_dists))
    pos_dists, lat_dists = pos_dists[:n], lat_dists[:n]

    mask = pos_dists > 1e-3
    pos_d = pos_dists[mask]
    lat_d = lat_dists[mask]

    lat_d = (lat_d - lat_d.min()) / (lat_d.max() - lat_d.min())

    p_xy = np.histogram2d(pos_d, lat_d, bins=10)[0]
    p_xy /= p_xy.sum()
    p_x = p_xy.sum(axis=0)
    p_y = p_xy.sum(axis=1)
    marginal = p_y[:, None] @ p_x[None, :]
    p_idx = (p_xy != 0) & (marginal != 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sum((p_xy * (np.log2(p_xy) - np.log2(marginal)))[p_idx])


def load_flat(rec, subdir):
    lat_path = Path(f'data/rec_{rec}/{subdir}/latents_flat.npy')
    pos_path = Path(f'data/rec_{rec}/{subdir}/positions.npy')
    if not lat_path.exists() or not pos_path.exists():
        return None, None
    return (np.load(lat_path).astype(np.float32),
            np.load(pos_path).astype(np.float32))


def load_place_field_binary(rec, subdir, bin_size_m=2.0):
    """Proposed mechanism (ACCM Figure 2e style): spatially-averaged binary vectors.

    Divides the floor into bin_size_m × bin_size_m grid cells. For each cell,
    averages the pooled latents of all visiting episodes → stable per-bin vector.
    Binarises per unit at the median across visited bins. Each episode gets the
    binary vector of its cell — spatially coherent, no 90th-percentile needed.
    """
    pool_path = Path(f'data/rec_{rec}/{subdir}/latents_pooled.npy')
    pos_path  = Path(f'data/rec_{rec}/{subdir}/positions.npy')
    if not pool_path.exists() or not pos_path.exists():
        return None, None

    pooled    = np.load(pool_path).astype(np.float32)   # [N, 128]
    positions = np.load(pos_path).astype(np.float32)    # [N, 2]

    xspan = positions[:, 0].max() - positions[:, 0].min()
    yspan = positions[:, 1].max() - positions[:, 1].min()
    n_bins_x = max(2, int(np.ceil(xspan / bin_size_m)))
    n_bins_y = max(2, int(np.ceil(yspan / bin_size_m)))

    x_edges = np.linspace(positions[:, 0].min(), positions[:, 0].max() + 1e-6, n_bins_x + 1)
    y_edges = np.linspace(positions[:, 1].min(), positions[:, 1].max() + 1e-6, n_bins_y + 1)
    bx = np.clip(np.searchsorted(x_edges, positions[:, 0], side='right') - 1, 0, n_bins_x - 1)
    by = np.clip(np.searchsorted(y_edges, positions[:, 1], side='right') - 1, 0, n_bins_y - 1)

    # Average pooled latents within each bin
    bin_means  = np.zeros((n_bins_x, n_bins_y, 128), dtype=np.float32)
    bin_counts = np.zeros((n_bins_x, n_bins_y), dtype=np.int32)
    np.add.at(bin_means,  (bx, by), pooled)
    np.add.at(bin_counts, (bx, by), 1)
    visited = bin_counts > 0
    bin_means[visited] /= bin_counts[visited, None]

    # Binarise per unit at median across visited bins
    visited_means = bin_means[visited]                           # [V, 128]
    thresh = np.median(visited_means, axis=0)                    # [128]
    bin_binary = np.zeros((n_bins_x, n_bins_y, 128), dtype=np.float32)
    bin_binary[visited] = (visited_means > thresh).astype(np.float32)

    binary = bin_binary[bx, by]                                  # [N, 128]
    return binary, positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recs',         nargs='+', default=None)
    parser.add_argument('--rec',          type=str,  default=None)
    parser.add_argument('--mode',         type=str,  default='depth',
                        choices=['depth', 'rgb', 'all'])
    parser.add_argument('--offset',       type=int,  default=13)
    parser.add_argument('--neighborhood', type=int,  default=50)
    parser.add_argument('--shuffle',      action='store_true',
                        help='Shuffle episodes before MI (replicates ACCM random-path sampling)')
    args = parser.parse_args()

    recs = args.recs if args.recs else ([args.rec] if args.rec else [f'{i:02d}' for i in range(12, 22)])
    recs = [str(r).zfill(2) for r in recs]

    print(f"\noffset={args.offset}  neighborhood={args.neighborhood}")

    col_w = 12

    if args.mode == 'all':
        header = f"{'Rec':<10}  {'Depth PC':>{col_w}}  {'Prop Mech':>{col_w}}  {'RGB PC':>{col_w}}"
        print(header)
        print("-" * len(header))
        d_mis, pm_mis, r_mis = [], [], []
        for rec in recs:
            d_lat, pos  = load_flat(rec, 'latents')
            r_lat, _    = load_flat(rec, 'latents_rgb')
            pm_lat, _   = load_place_field_binary(rec, 'latents')

            d  = f"{compute_mi(d_lat,  pos, args.offset, args.neighborhood, shuffle=args.shuffle):>{col_w}.4f}" if d_lat  is not None else f"{'N/A':>{col_w}}"
            pm = f"{compute_mi(pm_lat, pos, args.offset, args.neighborhood, shuffle=args.shuffle):>{col_w}.4f}" if pm_lat is not None else f"{'N/A':>{col_w}}"
            r  = f"{compute_mi(r_lat,  pos, args.offset, args.neighborhood, shuffle=args.shuffle):>{col_w}.4f}" if r_lat  is not None else f"{'N/A':>{col_w}}"
            print(f"rec_{rec:<6}  {d}  {pm}  {r}")

            if d_lat  is not None: d_mis.append(float(d))
            if pm_lat is not None: pm_mis.append(float(pm))
            if r_lat  is not None: r_mis.append(float(r))

        print(f"\n{'='*len(header)}")
        if d_mis:  print(f"Mean Depth PC:    {np.mean(d_mis):.4f} bits")
        if pm_mis: print(f"Mean Prop Mech:   {np.mean(pm_mis):.4f} bits")
        if r_mis:  print(f"Mean RGB PC:      {np.mean(r_mis):.4f} bits")
        print(f"ACCM PC:          0.627 bits  |  ACCM Prop Mech: 0.542 bits  |  ACCM AE: ~0.20 bits")

    else:
        subdir = 'latents' if args.mode == 'depth' else 'latents_rgb'
        lbl    = 'Depth PC' if args.mode == 'depth' else 'RGB PC'
        header = f"{'Rec':<10}  {lbl:>{col_w}}  {'Prop Mech':>{col_w}}"
        print(header)
        print("-" * len(header))
        mis, pm_mis = [], []
        for rec in recs:
            lat, pos    = load_flat(rec, subdir)
            pm_lat, _   = load_place_field_binary(rec, subdir)
            m  = f"{compute_mi(lat,    pos, args.offset, args.neighborhood, shuffle=args.shuffle):>{col_w}.4f}" if lat    is not None else f"{'N/A':>{col_w}}"
            pm = f"{compute_mi(pm_lat, pos, args.offset, args.neighborhood, shuffle=args.shuffle):>{col_w}.4f}" if pm_lat is not None else f"{'N/A':>{col_w}}"
            print(f"rec_{rec:<6}  {m}  {pm}")
            if lat    is not None: mis.append(float(m))
            if pm_lat is not None: pm_mis.append(float(pm))

        print(f"\n{'='*len(header)}")
        if mis:    print(f"Mean {lbl}:  {np.mean(mis):.4f} bits")
        if pm_mis: print(f"Mean Prop Mech: {np.mean(pm_mis):.4f} bits")
        print(f"ACCM PC: 0.627 bits  |  ACCM Prop Mech: 0.542 bits  |  ACCM AE: ~0.20 bits")


if __name__ == '__main__':
    main()
