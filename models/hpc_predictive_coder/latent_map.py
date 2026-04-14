#!/usr/bin/env python3
"""
Plot latent representations on real-world odometry map.

Two plots:
1. UMAP of latent space, coloured by physical position
2. Physical odometry map, coloured by UMAP embedding (dim 0 and dim 1)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latents', type=str, default='./rec_wh_inf_01/latents/latents_flat.npy')
    parser.add_argument('--positions', type=str, default='./rec_wh_inf_01/latents/positions.npy')
    parser.add_argument('--output', type=str, default='latent_map.png')
    parser.add_argument('--n_neighbors', type=int, default=15)
    parser.add_argument('--min_dist', type=float, default=0.1)
    args = parser.parse_args()

    print("Loading data...")
    latents = np.load(args.latents).astype(np.float32)
    positions = np.load(args.positions).astype(np.float32)
    x, y = positions[:, 0], positions[:, 1]

    print(f"  Latents:   {latents.shape}")
    print(f"  Positions: {positions.shape}")

    print("Running UMAP...")
    import umap
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                        n_components=2, random_state=42)
    embedding = reducer.fit_transform(latents)  # (N, 2)
    print(f"  Embedding: {embedding.shape}")

    u0 = embedding[:, 0]
    u1 = embedding[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: UMAP space coloured by x position ---
    ax = axes[0]
    sc = ax.scatter(u0, u1, c=x, s=2, alpha=0.5, cmap='RdYlBu')
    plt.colorbar(sc, ax=ax, label='x position (m)')
    ax.set_title("UMAP latent space (coloured by x)")
    ax.set_xlabel("UMAP dim 0")
    ax.set_ylabel("UMAP dim 1")

    # --- Plot 2: Odometry map coloured by UMAP dim 0 ---
    ax = axes[1]
    sc = ax.scatter(x, y, c=u0, s=4, alpha=0.6, cmap='plasma')
    plt.colorbar(sc, ax=ax, label='UMAP dim 0')
    ax.set_title("Odometry map (coloured by UMAP dim 0)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect('equal')

    # --- Plot 3: Odometry map coloured by UMAP dim 1 ---
    ax = axes[2]
    sc = ax.scatter(x, y, c=u1, s=4, alpha=0.6, cmap='plasma')
    plt.colorbar(sc, ax=ax, label='UMAP dim 1')
    ax.set_title("Odometry map (coloured by UMAP dim 1)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\nSaved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
