#!/usr/bin/env python3
"""Map evaluation (ACCM pairwise test), Set B.
Pearson = pearsonr(LOG physical distance, latent distance), exactly scripts/compare_pearson.py (ACCM's claim).
MI = scripts/mutual_information.compute_mi. Neighbourhood 60. Never raw distance (see memory).
Random-init floors belong ONLY here (--floor): mean over 5 seeds per recording, each trained model compared with
the floor of its own architecture. The TwoHead LC head is not a mapping head and is never reported here.

Examples:
  python3 scripts/eval/map_eval.py --input depth --stride 10
  python3 scripts/eval/map_eval.py --input rgb --models "VPC-RGB-{fed}-fp32" "TwoHead-RGB-{fed}-bf16" --floor
  python3 scripts/eval/map_eval.py --input depth --stride 20 --compare "TwoHead-RGB-Depth-bf16 MAP" "VPC-RGB-Depth-fp32"
Writes results/map_<input>_s<stride>.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SET_B, RECSETS, short, NEIGHBORHOOD, offset, get_latents, full_name, REGISTRY
from compare_pearson import pearson_for_recording
from mutual_information import compute_mi

DEFAULT = ['VPC-Depth-{fed}-fp32', 'VPC-RGB-{fed}-fp32', 'AE-Depth-{fed}-fp32', 'AE-RGB-{fed}-fp32', 'CPC-Depth-{fed}-fp32',
           'JEPATR-Depth-{fed}-fp32', 'JEVPC-Depth-{fed}-fp32', 'JEVPC-RGB-{fed}-fp32', 'TwoHead-RGB-{fed}-fp32',
           'TwoHead-RGB-{fed}-bf16', 'VJEPA-Hybrid-RGB-{fed}-bf16']
FLOOR_ARCH = {'vpc': 'Random-VPC-arch', 'twohead': 'Random-TwoHead-arch', 'vjepa': 'Random-VJEPA-arch'}


def floor_arch(stem):
    """Random-init floor matching the stem's architecture (V-JEPA has two encoders: hybrid and ViT)."""
    k = REGISTRY[stem]['kind']
    if k == 'vjepa' and 'ViT' in stem:
        return 'Random-VJEPAViT-arch'
    return FLOOR_ARCH.get(k)


def score(lat, pos, rec, stride):
    tmp = ROOT / 'results/.tmp'; tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / 'l.npy', lat); np.save(tmp / 'p.npy', pos)
    off = offset(rec, stride, stride)
    return (pearson_for_recording(tmp / 'l.npy', tmp / 'p.npy', off, NEIGHBORHOOD)[0],
            compute_mi(lat, pos, off, NEIGHBORHOOD, min(10000, len(lat) - off)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', choices=['depth', 'rgb'], default='depth')
    ap.add_argument('--recs', choices=['set_b', 'scannet'], default='set_b')
    ap.add_argument('--stride', type=int, default=10, help='stride = seq_len (10 or 20)')
    ap.add_argument('--models', nargs='+', default=DEFAULT)
    ap.add_argument('--floor', action='store_true', help='add 5-seed random-init floors for the architectures present')
    ap.add_argument('--compare', nargs=2, action='append', default=[], metavar=('A', 'B'), help='extra paired test A vs B')
    a = ap.parse_args()
    RECS = RECSETS[a.recs]
    fed = 'Depth' if a.input == 'depth' else 'RGB'
    stems = [m for m in a.models if REGISTRY[m]['ch'] == 1 and a.input == 'depth' or REGISTRY[m]['ch'] == 3]
    floors = {}
    if a.floor:
        for m in stems:
            arch = floor_arch(m)
            if arch:
                floors[arch] = [f'{arch}-s{s}' for s in range(5)]
    lats = get_latents(stems + [s for v in floors.values() for s in v], RECS, a.input, a.stride)

    res = {}
    for name, per_rec in lats.items():
        if name.endswith(' LC'):
            continue                                                        # LC head is not a mapping head
        res[name] = np.array([score(*per_rec[r], r, a.stride) for r in RECS])
    for arch, seeds in floors.items():                                        # floor = per-rec mean over seeds
        keys = [s + (' MAP' if 'TwoHead' in arch else '') for s in seeds]
        stack = np.stack([res.pop(k) for k in keys])
        res[f'{arch} (5-seed floor)'] = stack.mean(0)
        res[f'{arch} (5-seed floor)_sd'] = stack.mean(1).std(0)

    lines = [f"Map eval, {a.recs} ({len(RECS)} recordings), input={a.input}, stride={a.stride}/seq_len={a.stride}, Pearson = log distance"]
    for j, metric in enumerate(('Pearson (log)', 'MI')):
        lines.append(f"\n#### {metric} ####")
        lines.append(f"{'model':38s}" + "".join(f"{short(r):>8s}" for r in RECS) + f"{'MEAN':>8s}")
        for k, v in res.items():
            if k.endswith('_sd'):
                continue
            sd = f"  ±{res[k + '_sd'][j]:.4f} (seed SD)" if k + '_sd' in res else ''
            lines.append(f"{k:38s}" + "".join(f"{x:7.3f}" for x in v[:, j]) + f"{v[:, j].mean():8.4f}{sd}")

    pairs = [tuple(p) for p in a.compare]
    for stem in stems:                                                        # each trained model vs its own floor
        arch = floor_arch(stem)
        if arch and arch in floors:
            pairs.append((full_name(stem, fed) + (' MAP' if REGISTRY[stem]['kind'] == 'twohead' else ''), f'{arch} (5-seed floor)'))
    if pairs:
        lines.append(f"\n#### paired tests over the {len(RECS)} recordings ####")
    for x, y in pairs:
        s = []
        for j, m in enumerate(('Pearson', 'MI')):
            u, v = res[x][:, j], res[y][:, j]
            s.append(f"{m} {(u - v).mean():+.4f} wins {(u > v).sum()}/{len(u)} (t p={ttest_rel(u, v).pvalue:.3g}, W p={wilcoxon(u, v).pvalue:.3g})")
        lines.append(f"{x} vs {y}:\n    " + " | ".join(s))

    text = "\n".join(lines); print(text)
    out = ROOT / 'results' / (f'map_{a.input}_s{a.stride}' + ('' if a.recs == 'set_b' else f'_{a.recs}') + '.txt'); out.write_text(text + "\n")
    print(f"\n-> {out}")


if __name__ == '__main__':
    main()
