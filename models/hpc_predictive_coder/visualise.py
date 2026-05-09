#!/usr/bin/env python3
"""
Visualise HPCPredictiveCoder predictions.

Shows: input t | predicted t+1 | ground truth t+1
Supports depth (.npy) and RGB (.jpg/.png) via --mode flag.
"""

import argparse
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import create_hpc_predictive_coder


def load_depth_frame(path, max_depth=20.0):
    depth = np.load(path).astype(np.float32)
    if depth.ndim == 2:
        depth = depth[np.newaxis]
    depth = np.where(np.isinf(depth) | np.isnan(depth), max_depth, depth)
    depth = np.clip(depth, 0, max_depth) / max_depth
    return depth  # (1, H, W)


def load_rgb_frame(path):
    from PIL import Image
    img = np.array(Image.open(path)).astype(np.float32) / 255.0  # (H, W, 3)
    return img.transpose(2, 0, 1)  # (3, H, W)


def to_img(t, mode):
    arr = t.squeeze().cpu().numpy()
    if mode == 'rgb':
        return np.clip(arr.transpose(1, 2, 0), 0, 1)  # (H, W, 3)
    return arr  # (H, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--mode', type=str, default='depth', choices=['depth', 'rgb'],
                        help='Input modality')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--max_depth', type=float, default=20.0)
    parser.add_argument('--fps', type=float, default=55.0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('-n', type=int, default=None,
                        help='Sample N random windows instead of sequential playback')
    args = parser.parse_args()

    is_rgb = args.mode == 'rgb'
    in_channels = 3 if is_rgb else 1
    load_frame = load_rgb_frame if is_rgb else lambda p: load_depth_frame(p, args.max_depth)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = create_hpc_predictive_coder(
        seq_len=args.seq_len,
        in_channels=in_channels,
        out_channels=in_channels,
        latent_dim=args.latent_dim
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")

    # Find files
    if is_rgb:
        exts = ['*.jpg', '*.jpeg', '*.png']
        files = []
        for ext in exts:
            files.extend(sorted(Path(args.data_path).glob(ext)))
        files = sorted(files)
    else:
        files = sorted(Path(args.data_path).glob("*.npy"))

    if len(files) < args.seq_len + 1:
        raise ValueError(f"Need at least {args.seq_len + 1} frames, got {len(files)}")
    print(f"Found {len(files)} {args.mode} frames")

    # Setup figure
    cmap = None if is_rgb else 'inferno'
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, title in zip(axes, ['Input  t', 'Predicted  t+1', 'GT  t+1']):
        ax.set_title(title)
        ax.set_axis_off()

    start = max(args.start, 0)
    init_window = [load_frame(files[i]) for i in range(start, start + args.seq_len)]
    seq = torch.from_numpy(np.stack(init_window, axis=0)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(seq)

    init_kwargs = dict(vmin=0, vmax=1) if not is_rgb else {}
    imgs = [
        axes[0].imshow(to_img(seq[0, -1], args.mode), cmap=cmap, **init_kwargs),
        axes[1].imshow(to_img(pred[0, -1], args.mode), cmap=cmap, **init_kwargs),
        axes[2].imshow(to_img(seq[0, -1], args.mode), cmap=cmap, **init_kwargs),
    ]
    if not is_rgb:
        for img, ax in zip(imgs, axes):
            plt.colorbar(img, ax=ax)

    frame_title = fig.suptitle('')
    plt.tight_layout()

    interval = 1.0 / args.fps
    max_i = len(files) - 1
    sequential = range(start + args.seq_len - 1, max_i)
    if args.n is not None:
        indices = sorted(np.random.choice(list(sequential), size=min(args.n, len(sequential)), replace=False))
    else:
        indices = list(sequential)

    for i in indices:
        window = [load_frame(files[j]) for j in range(max(0, i - args.seq_len + 1), i + 1)]
        seq = torch.from_numpy(np.stack(window, axis=0)).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(seq)

        t_frame    = to_img(seq[0, -2], args.mode)
        pred_frame = to_img(pred[0, -2], args.mode)
        gt_frame   = to_img(seq[0, -1], args.mode)

        err = np.abs(pred_frame - gt_frame).mean()
        abs_m = err * args.max_depth if not is_rgb else err

        imgs[0].set_data(t_frame)
        imgs[1].set_data(pred_frame)
        imgs[2].set_data(gt_frame)

        label = f"Frame {i}/{len(files)-1}  |  MAE: {err:.4f}"
        if not is_rgb:
            label += f"  |  Abs: {abs_m:.3f}m"
        frame_title.set_text(label)
        plt.pause(interval)

    plt.show()


if __name__ == "__main__":
    main()
