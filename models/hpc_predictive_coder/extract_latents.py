#!/usr/bin/env python3
"""
Extract latents from HPCPredictiveCoder using sliding window of seq_len frames.
Uses model.get_latents() which includes temporal attention.

Supports depth (.npy) and RGB (.jpg/.png) via --mode flag.

Output:
  - latents_pooled.npy  [N, latent_dim]      (pooled over spatial dims)
  - latents_flat.npy    [N, latent_dim*H*W]  (flattened spatial features)
  - positions.npy       [N, 2]               (x, y from odometry, aligned)
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model import create_hpc_predictive_coder


def load_depth_frames(depth_dir, max_depth=20.0):
    depth_dir = Path(depth_dir)
    files = sorted(depth_dir.glob("*.npy"))
    print(f"Found {len(files)} depth frames in {depth_dir}")
    frames = []
    for f in tqdm(files, desc="Loading depth frames"):
        depth = np.load(f).astype(np.float32)
        depth = np.where(np.isinf(depth) | np.isnan(depth), max_depth, depth)
        depth = np.clip(depth, 0, max_depth) / max_depth
        frames.append(depth)
    return np.array(frames)  # [T, H, W]


def load_rgb_frames(rgb_dir):
    rgb_dir = Path(rgb_dir)
    files = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
    files = sorted(files)
    print(f"Found {len(files)} RGB frames in {rgb_dir}")
    frames = []
    for f in tqdm(files, desc="Loading RGB frames"):
        img = np.array(Image.open(f)).astype(np.float32) / 255.0  # (H, W, 3)
        frames.append(img.transpose(2, 0, 1))  # (3, H, W)
    return np.array(frames)  # [T, 3, H, W]


def extract_latents(model, frames, seq_len=10, device='cuda', is_rgb=False, stride=1, step=1):
    model.eval()
    T = len(frames)
    # step > 1 means we sample every `step` frames within the window,
    # so the actual frame span is seq_len * step
    span = seq_len * step
    starts = range(0, T - span + 1, stride)
    N = len(starts)

    latents_pooled = []
    latents_flat = []

    print(f"\nExtracting latents: {N} windows (stride={stride}, seq_len={seq_len}, step={step}, span={span})")

    with torch.no_grad():
        for i in tqdm(starts, desc="Extracting"):
            window = frames[i : i + span : step]  # [seq_len, (C,) H, W]
            window_tensor = torch.from_numpy(window).float()

            if is_rgb:
                # window_tensor: [seq_len, 3, H, W] -> [1, seq_len, 3, H, W]
                window_tensor = window_tensor.unsqueeze(0).to(device)
            else:
                # window_tensor: [seq_len, H, W] -> [1, seq_len, 1, H, W]
                window_tensor = window_tensor.unsqueeze(0).unsqueeze(2).to(device)

            z = model.get_latents(window_tensor)  # [1, seq_len, C, H, W]
            z_last = z[0, -1]  # [C, H, W]

            z_pooled = z_last.mean(dim=[1, 2]).cpu().numpy()  # [C]
            z_flat = z_last.reshape(-1).cpu().numpy()          # [C*H*W]

            latents_pooled.append(z_pooled)
            latents_flat.append(z_flat)

    return np.array(latents_pooled), np.array(latents_flat)


def main():
    parser = argparse.ArgumentParser(description="Extract latents from HPCPredictiveCoder")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--depth_dir', type=str, default=None,
                        help='Path to depth .npy directory')
    parser.add_argument('--rgb_dir', type=str, default=None,
                        help='Path to RGB image directory')
    parser.add_argument('--odom_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--step',   type=int, default=1,
                        help='Frame step within window (step=2 → every 2nd frame, doubles temporal span)')
    parser.add_argument('--max_depth', type=float, default=20.0)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=4)
    args = parser.parse_args()

    if args.depth_dir is None and args.rgb_dir is None:
        raise ValueError("Provide either --depth_dir or --rgb_dir")

    is_rgb = args.rgb_dir is not None
    in_channels = 3 if is_rgb else 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\n1. Loading model from {args.checkpoint}")
    model = create_hpc_predictive_coder(
        seq_len=args.seq_len,
        in_channels=in_channels,
        out_channels=in_channels,
        num_heads=args.num_heads,
        latent_dim=args.latent_dim
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    print(f"   Loaded epoch {checkpoint.get('epoch', '?')}, val_loss={checkpoint.get('val_loss', '?')}")

    print(f"\n2. Loading frames")
    if is_rgb:
        frames = load_rgb_frames(args.rgb_dir)
    else:
        frames = load_depth_frames(args.depth_dir, args.max_depth)
    print(f"   Loaded: {frames.shape}")

    print(f"\n3. Loading odometry from {args.odom_file}")
    odom = np.load(args.odom_file)
    positions_all = np.stack([odom['x'], odom['y']], axis=1).astype(np.float32)
    print(f"   Loaded: {positions_all.shape}")

    assert len(frames) == len(positions_all), \
        f"Mismatch: {len(frames)} frames vs {len(positions_all)} odom entries"

    print(f"\n4. Extracting latents")
    latents_pooled, latents_flat = extract_latents(
        model, frames, args.seq_len, device, is_rgb, args.stride, args.step
    )

    span = args.seq_len * args.step
    positions = positions_all[[s + span - 1 for s in range(0, len(positions_all) - span + 1, args.stride)]]
    assert len(latents_pooled) == len(positions), \
        f"Mismatch: {len(latents_pooled)} latents vs {len(positions)} positions"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'latents_pooled.npy', latents_pooled)
    np.save(output_dir / 'latents_flat.npy', latents_flat)
    np.save(output_dir / 'positions.npy', positions)

    print(f"\n5. Saved to {output_dir}/")
    print(f"   latents_pooled.npy: {latents_pooled.shape}")
    print(f"   latents_flat.npy:   {latents_flat.shape}")
    print(f"   positions.npy:      {positions.shape}")
    print(f"\n   Position range: x=[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}], "
          f"y=[{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]")


if __name__ == "__main__":
    main()
