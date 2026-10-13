#!/usr/bin/env python3
"""
Extract BOTH heads' latents from JEVPC-TwoHead, sliding window, same
conventions as jevpc/extract_latents_jevpc.py (last-position output of each
window). Writes <output_dir>/map/ and <output_dir>/lc/, each with
latents_pooled.npy, latents_flat.npy, positions.npy.

--depth_as_3ch: feed depth broadcast to 3 channels into an RGB checkpoint.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from jevpc_twohead import JEVPCTwoHead


def load_depth(d, max_depth=20.0, broadcast_3ch=False):
    files = sorted(Path(d).glob("*.npy"))
    out = []
    for f in tqdm(files, desc="depth"):
        x = np.load(f).astype(np.float32)
        x = np.clip(np.where(np.isinf(x) | np.isnan(x), max_depth, x), 0, max_depth) / max_depth
        out.append(np.repeat(x[np.newaxis], 3, axis=0) if broadcast_3ch else x[np.newaxis])
    return np.array(out)


def load_rgb(d):
    files = sorted(list(Path(d).glob("*.jpg")) + list(Path(d).glob("*.png")))
    return np.array([(np.array(Image.open(f)).astype(np.float32) / 255.0).transpose(2, 0, 1)
                     for f in tqdm(files, desc="rgb")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--depth_dir')
    ap.add_argument('--rgb_dir')
    ap.add_argument('--depth_as_3ch', action='store_true')
    ap.add_argument('--odom_file', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--stride', type=int, default=10)
    a = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model = JEVPCTwoHead(in_channels=ck['in_channels'], latent_dim=ck['latent_dim'], num_heads=ck['num_heads'])
    model.load_state_dict(ck['model_state_dict'])
    model = model.to(device).eval()
    print(f"Loaded epoch {ck['epoch']}, val_loss={ck['val_loss']:.6f}, in_channels={ck['in_channels']}")

    if a.rgb_dir:
        frames = load_rgb(a.rgb_dir)
    else:
        frames = load_depth(a.depth_dir, broadcast_3ch=a.depth_as_3ch)
    assert frames.shape[1] == ck['in_channels'], f"input has {frames.shape[1]} ch, model expects {ck['in_channels']}"

    odom = np.load(a.odom_file)
    pos_all = np.stack([odom['x'], odom['y']], axis=1).astype(np.float32)
    assert len(frames) == len(pos_all), f"{len(frames)} frames vs {len(pos_all)} odom"

    starts = list(range(0, len(frames) - a.seq_len + 1, a.stride))
    res = {'map': ([], []), 'lc': ([], [])}
    with torch.no_grad():
        for s in tqdm(starts, desc="extract"):
            x = torch.from_numpy(frames[s:s + a.seq_len]).float().unsqueeze(0).to(device)
            h_map, h_lc = model.get_latents_both(x)
            for k, h in (('map', h_map), ('lc', h_lc)):
                last = h[0, -1]
                res[k][0].append(last.mean(dim=[1, 2]).cpu().numpy())
                res[k][1].append(last.reshape(-1).cpu().numpy())

    positions = pos_all[[s + a.seq_len - 1 for s in starts]]
    for k, (pooled, flat) in res.items():
        d = Path(a.output_dir) / k
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / 'latents_pooled.npy', np.array(pooled))
        np.save(d / 'latents_flat.npy', np.array(flat))
        np.save(d / 'positions.npy', positions)
        print(f"{k}: latents_flat {np.array(flat).shape} -> {d}")


if __name__ == "__main__":
    main()
