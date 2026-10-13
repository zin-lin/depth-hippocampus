#!/usr/bin/env python3
"""
Extract latents from HPCPredictiveCoder trained on ACCM's village dataset.

Loads all 8263 val-dataset episodes (10 frames each), runs each through the
model as a single sequence, and saves the last-frame latent + position.

Output (in --output_dir):
    latents_pooled.npy  [N, 128]       spatially mean-pooled
    latents_flat.npy    [N, 128*8*8]   full spatial latent
    positions.npy       [N, 2]         (x, y) from last frame of each episode

Usage:
    python3 models/hpc_predictive_coder/accm_analysis/extract_accm_latents.py \
        --checkpoint weights/accm_rgb/best.pth \
        --val_dir    data/accm/datasets/val-dataset \
        --output_dir data/accm/latents
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "vpc"))
from model import create_hpc_predictive_coder

ACCM_MEAN = [121.6697 / 255, 149.3242 / 255, 154.9510 / 255]
ACCM_STD  = [40.7521  / 255, 47.7267  / 255, 103.2739 / 255]

IMG_EXTS = ('.png', '.PNG', '.jpg', '.jpeg')


def load_episode(epi_dir: Path, transform) -> torch.Tensor:
    """Load all frames from an episode folder, sorted lexicographically (matches ACCM)."""
    fns = sorted([f for f in epi_dir.iterdir() if f.suffix in IMG_EXTS], key=lambda f: f.name)
    return torch.stack([transform(Image.open(f)) for f in fns], dim=0)  # [L, 3, H, W]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--val_dir',     type=str, required=True)
    parser.add_argument('--output_dir',  type=str, required=True)
    parser.add_argument('--latent_dim',  type=int, default=128)
    parser.add_argument('--num_heads',   type=int, default=4)
    parser.add_argument('--seq_len',     type=int, default=10,
                        help='Frames per val episode (default 10)')
    parser.add_argument('--batch_size',  type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(ACCM_MEAN, ACCM_STD),
    ])

    print(f"\n1. Loading model from {args.checkpoint}")
    model = create_hpc_predictive_coder(
        seq_len=args.seq_len,
        in_channels=3,
        out_channels=3,
        num_heads=args.num_heads,
        latent_dim=args.latent_dim,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    print(f"   Loaded epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")

    print(f"\n2. Scanning val episodes in {args.val_dir}")
    episodes = sorted([p for p in Path(args.val_dir).glob('*') if p.is_dir()])
    print(f"   Found {len(episodes)} episodes")

    print(f"\n3. Extracting latents (batch_size={args.batch_size})")
    print("   Using ALL frames per episode (matching ACCM: N_episodes × L frames each)")
    latents_pooled = []
    latents_flat   = []
    positions      = []

    with torch.no_grad():
        for i in tqdm(range(0, len(episodes), args.batch_size), desc="Batches"):
            batch_epis = episodes[i : i + args.batch_size]
            B = len(batch_epis)

            seqs = torch.stack([load_episode(e, transform) for e in batch_epis], dim=0)
            # [B, L, 3, H, W]
            L = seqs.shape[1]
            seqs = seqs.to(device)

            z = model.get_latents(seqs)          # [B, L, C, H, W]
            z_all = z.reshape(B * L, *z.shape[2:])  # [B*L, C, H, W]

            z_pooled = z_all.mean(dim=[2, 3]).cpu().numpy()       # [B*L, C]
            z_flat   = z_all.reshape(B * L, -1).cpu().numpy()     # [B*L, C*H*W]

            latents_pooled.append(z_pooled)
            latents_flat.append(z_flat)

            # ALL frame (x, y) positions from state.npz
            for epi in batch_epis:
                state = np.load(epi / 'state.npz')['arr_0']  # [L, 3]
                positions.append(state[:, :2])                # [L, 2]

    latents_pooled = np.concatenate(latents_pooled, axis=0).astype(np.float32)
    latents_flat   = np.concatenate(latents_flat,   axis=0).astype(np.float32)
    positions      = np.concatenate(positions,      axis=0).astype(np.float32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'latents_pooled.npy', latents_pooled)
    np.save(output_dir / 'latents_flat.npy',   latents_flat)
    np.save(output_dir / 'positions.npy',       positions)

    print(f"\n4. Saved to {output_dir}/")
    print(f"   latents_pooled.npy : {latents_pooled.shape}")
    print(f"   latents_flat.npy   : {latents_flat.shape}")
    print(f"   positions.npy      : {positions.shape}")
    print(f"   Position range: x=[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}], "
          f"y=[{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]")


if __name__ == '__main__':
    main()
