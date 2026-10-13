#!/usr/bin/env python3
"""Convert ScanNet .sens recordings into our Gazebo recording layout (64x64 depth + RGB + floor-plane positions).

ScanNet DATA is under the ScanNet Terms of Use (research only, NO redistribution): everything this script writes
stays on /mnt/lexar (outside the repo, never committed or published). This script contains no ScanNet data; the .sens
reader is written from the published format description (header, then per frame: camera-to-world 4x4, timestamps,
JPEG colour bytes, zlib uint16 depth bytes).

Output per scene (same conventions as data/rec_*):
  <out>/<scene>/depth64/%06d.npy   float32 metres, 64x64; invalid (0) -> 20.0 (= our "no return" max range)
  <out>/<scene>/images64/%06d.jpg  RGB 64x64
  <out>/<scene>/odometry.npz       x, y floor-plane positions (ScanNet world is z-up); also odometry_i.npz (same frames)
Every --step'th frame is kept (default 2: ScanNet moves ~1-1.5 cm/frame vs Gazebo ~2.5). Frames with an invalid pose (non-finite camera-to-world) are dropped. Square centre-crop before resizing (64x64).
Depth uses nearest-neighbour resizing so invalid pixels never blend with valid ones.

Usage: python3 scripts/scannet_convert.py --scans /mnt/lexar/scannet/scans --out /mnt/lexar/scannet_64 [--ids scene0011_00 ...]
"""
import argparse
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np

MAX_DEPTH = 20.0


def read_sens(path):
    with open(path, 'rb') as f:
        version = struct.unpack('I', f.read(4))[0]
        assert version == 4, f'{path}: unsupported .sens version {version}'
        f.read(struct.unpack('Q', f.read(8))[0])                        # sensor name
        f.read(64 * 4)                                                  # 4 intrinsic/extrinsic matrices
        col_comp, dep_comp = struct.unpack('ii', f.read(8))
        assert col_comp == 2 and dep_comp == 1, f'{path}: expected JPEG colour + zlib depth, got {col_comp}, {dep_comp}'
        cw, ch, dw, dh = struct.unpack('IIII', f.read(16))
        shift = struct.unpack('f', f.read(4))[0]
        nf = struct.unpack('Q', f.read(8))[0]
        for _ in range(nf):
            c2w = np.frombuffer(f.read(64), np.float32).reshape(4, 4)
            f.read(16)                                                  # colour + depth timestamps
            cs, ds = struct.unpack('QQ', f.read(16))
            cbytes, dbytes = f.read(cs), f.read(ds)
            yield c2w, cbytes, dbytes, (dh, dw), shift


def centre_square(a):
    h, w = a.shape[:2]; s = min(h, w); t, l = (h - s) // 2, (w - s) // 2
    return a[t:t + s, l:l + s]


def convert(sens, out, step=1):
    scene = sens.stem
    dd, id_ = out / scene / 'depth64', out / scene / 'images64'
    dd.mkdir(parents=True, exist_ok=True); id_.mkdir(parents=True, exist_ok=True)
    xs, ys, k, dropped = [], [], 0, 0
    for fi, (c2w, cb, db, (dh, dw), shift) in enumerate(read_sens(sens)):
        if fi % step:
            continue
        if not np.all(np.isfinite(c2w)):
            dropped += 1; continue
        depth = np.frombuffer(zlib.decompress(db), np.uint16).reshape(dh, dw).astype(np.float32) / shift
        depth = cv2.resize(centre_square(depth), (64, 64), interpolation=cv2.INTER_NEAREST)
        depth[depth <= 0] = MAX_DEPTH
        rgb = cv2.imdecode(np.frombuffer(cb, np.uint8), cv2.IMREAD_COLOR)          # BGR
        rgb = cv2.resize(centre_square(rgb), (64, 64), interpolation=cv2.INTER_AREA)
        np.save(dd / f'{k:06d}.npy', np.clip(depth, 0, MAX_DEPTH).astype(np.float32))
        cv2.imwrite(str(id_ / f'{k:06d}.jpg'), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        xs.append(float(c2w[0, 3])); ys.append(float(c2w[1, 3])); k += 1
    x, y = np.array(xs, np.float32), np.array(ys, np.float32)
    np.savez(out / scene / 'odometry.npz', x=x, y=y)
    np.savez(out / scene / 'odometry_i.npz', x=x, y=y)
    path_len = float(np.linalg.norm(np.diff(np.stack([x, y], 1), axis=0), axis=1).sum()) if k > 1 else 0.0
    print(f'{scene}: {k} frames kept, {dropped} dropped (invalid pose), path {path_len:.1f} m, '
          f'{path_len / max(1, k - 1) * 100:.2f} cm/frame', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scans', default='/mnt/lexar/scannet/scans')
    ap.add_argument('--out', default='/mnt/lexar/scannet_64')
    ap.add_argument('--ids', nargs='*', default=None)
    ap.add_argument('--step', type=int, default=2, help='keep every Nth frame (2 -> ~2-3 cm/frame, matching Gazebo ~2.5)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.scans).glob('*/*.sens'))
    if a.ids:
        files = [f for f in files if f.stem in a.ids]
    for f in files:
        if (out / f.stem / 'odometry.npz').exists():
            print(f'{f.stem}: already converted, skipping'); continue
        convert(f, out, a.step)


if __name__ == '__main__':
    main()
