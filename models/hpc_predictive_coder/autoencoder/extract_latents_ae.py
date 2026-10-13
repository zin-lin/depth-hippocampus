#!/usr/bin/env python3
"""
Extract latents from HPCAutoencoder for the downstream Pearson evaluation.

The autoencoder has no temporal attention, so a window's latent is simply the
encoder output of the window's LAST frame. Windows/positions are laid out
exactly as in vpc/extract_latents.py (starts = range(0, T - seq_len + 1, stride),
position = positions[start + seq_len - 1]) so results line up with the other
models at the same seq_len/stride.

Output:
  - latents_pooled.npy  [N, latent_dim]
  - latents_flat.npy    [N, latent_dim*H*W]
  - positions.npy       [N, 2]
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from autoencoder import HPCAutoencoder


def load_frame(f, is_rgb, max_depth):
    if is_rgb:
        img = np.array(Image.open(f)).astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)
    d = np.load(f).astype(np.float32)
    d = np.where(np.isinf(d) | np.isnan(d), max_depth, d)
    d = np.clip(d, 0, max_depth) / max_depth
    return d[np.newaxis]


def main():
    parser = argparse.ArgumentParser(description="Extract latents from HPCAutoencoder")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--depth_dir', type=str, default=None)
    parser.add_argument('--rgb_dir', type=str, default=None)
    parser.add_argument('--odom_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--max_depth', type=float, default=20.0)
    parser.add_argument('--latent_dim', type=int, default=128)
    args = parser.parse_args()

    if args.depth_dir is None and args.rgb_dir is None:
        raise ValueError("Provide either --depth_dir or --rgb_dir")

    is_rgb = args.rgb_dir is not None
    in_channels = 3 if is_rgb else 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HPCAutoencoder(in_channels, args.latent_dim)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    print(f"Loaded epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')}")

    if is_rgb:
        files = sorted(list(Path(args.rgb_dir).glob("*.jpg")) + list(Path(args.rgb_dir).glob("*.png")))
    else:
        files = sorted(Path(args.depth_dir).glob("*.npy"))

    odom = np.load(args.odom_file)
    positions_all = np.stack([odom['x'], odom['y']], axis=1).astype(np.float32)
    assert len(files) == len(positions_all), \
        f"Mismatch: {len(files)} frames vs {len(positions_all)} odom entries"

    last_idx = [s + args.seq_len - 1 for s in range(0, len(files) - args.seq_len + 1, args.stride)]

    pooled, flat = [], []
    with torch.no_grad():
        for i in range(0, len(last_idx), 256):
            batch_idx = last_idx[i:i + 256]
            x = torch.from_numpy(np.stack([load_frame(files[j], is_rgb, args.max_depth) for j in batch_idx])).float().to(device)
            z = model.encode(x)  # (B, C, h, w)
            pooled.append(z.mean(dim=[2, 3]).cpu().numpy())
            flat.append(z.reshape(z.shape[0], -1).cpu().numpy())

    latents_pooled = np.concatenate(pooled)
    latents_flat = np.concatenate(flat)
    positions = positions_all[last_idx]
    assert len(latents_pooled) == len(positions)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'latents_pooled.npy', latents_pooled)
    np.save(out / 'latents_flat.npy', latents_flat)
    np.save(out / 'positions.npy', positions)
    print(f"Saved {latents_flat.shape} latents, {positions.shape} positions to {out}")


if __name__ == "__main__":
    main()
