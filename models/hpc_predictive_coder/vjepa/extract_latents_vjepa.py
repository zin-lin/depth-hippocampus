#!/usr/bin/env python3
"""
Extract V-JEPA latents, sliding window, same conventions as the other extractors: the
target-encoder tokens of the LAST time step of each window -> (128, 8, 8) -> latents_flat 8192,
latents_pooled = 8x8-grid mean. Writes latents_pooled.npy, latents_flat.npy, positions.npy.

--depth_as_3ch: feed depth broadcast to 3 channels into an RGB checkpoint.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from vjepa import VJEPA

sys.path.insert(0, str(Path(__file__).parent.parent / 'jevpc_twohead'))
from extract_latents_twohead import load_depth, load_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--depth_dir')
    ap.add_argument('--rgb_dir')
    ap.add_argument('--depth_as_3ch', action='store_true')
    ap.add_argument('--odom_file', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--batch', type=int, default=64)
    a = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model = VJEPA(in_channels=ck['in_channels'], dim=ck['dim'], depth=ck['depth'], heads=ck['heads'],
                  pred_dim=ck['pred_dim'], pred_depth=ck['pred_depth'], pred_heads=ck['pred_heads'],
                  frames=ck['seq_len'], encoder=ck['encoder'])
    model.load_state_dict(ck['model_state_dict'])
    model = model.to(device).eval()
    seq_len = ck['seq_len']
    print(f"Loaded epoch {ck['epoch'] + 1}, val_loss={ck['val_loss']:.6f}, encoder={ck['encoder']}")

    frames = load_rgb(a.rgb_dir) if a.rgb_dir else load_depth(a.depth_dir, broadcast_3ch=a.depth_as_3ch)
    assert frames.shape[1] == ck['in_channels'], f"input has {frames.shape[1]} ch, model expects {ck['in_channels']}"

    odom = np.load(a.odom_file)
    pos_all = np.stack([odom['x'], odom['y']], axis=1).astype(np.float32)
    assert len(frames) == len(pos_all), f"{len(frames)} frames vs {len(pos_all)} odom"

    starts = list(range(0, len(frames) - seq_len + 1, a.stride))
    pooled, flat = [], []
    with torch.no_grad():
        for b in tqdm(range(0, len(starts), a.batch), desc="extract"):
            x = torch.from_numpy(np.stack([frames[s:s + seq_len] for s in starts[b:b + a.batch]])).float().to(device)
            last = model.get_latents(x)[:, -1]                 # (B, 128, 8, 8)
            pooled.append(last.mean(dim=[2, 3]).cpu().numpy())
            flat.append(last.reshape(len(x), -1).cpu().numpy())

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'latents_pooled.npy', np.concatenate(pooled))
    np.save(out / 'latents_flat.npy', np.concatenate(flat))
    np.save(out / 'positions.npy', pos_all[[s + seq_len - 1 for s in starts]])
    print(f"latents_flat {np.concatenate(flat).shape} -> {out}")


if __name__ == "__main__":
    main()
