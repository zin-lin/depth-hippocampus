#!/usr/bin/env python3
"""Perturbation test (user-approved name). Set B, stride 10.
Every model sees the SAME perturbed frames (seeded per recording + perturbation).
  map retention = perturbed / clean Pearson(log) and MI   (mean of per-recording ratios)
  latent shift  = median ||z_pert - z_clean|| / median clean random-pair distance (lower = more stable)
Trained MAP-type outputs only: no random-init networks, no TwoHead LC head (memory: feedback_random_floor_scope).

Depth perturbations ([0,1] normalised depth): noise s=.05 | blur s=1.5px | occlude 16x16 patch @.05 (near obstacle)
  | dropout 10% px -> max depth (missing returns) | scale x0.9 (calibration bias)
RGB perturbations: noise | blur | occlude (16x16 @ .5 grey) | dropout (10% px -> 1.0) | dim x0.7 | tint (+.1 red, -.1 blue)
For augmentation-trained models, noise + dropout were seen in training; the rest are held out.

Examples:
  python3 scripts/eval/perturb_test.py --input depth
  python3 scripts/eval/perturb_test.py --input depth --models "VPC-RGB-{fed}-fp32" "TwoHead-RGB-{fed}-bf16" "TwoHead-RGB-{fed}-bf16-aug"
Writes results/perturb_<input>.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SET_B, RECSETS, short, REGISTRY, load_model, load_frames, extract, full_name, heads
from map_eval import score

DEFAULT = ['VPC-Depth-{fed}-fp32', 'VPC-RGB-{fed}-fp32', 'AE-Depth-{fed}-fp32', 'AE-RGB-{fed}-fp32', 'CPC-Depth-{fed}-fp32',
           'JEPATR-Depth-{fed}-fp32', 'JEVPC-Depth-{fed}-fp32', 'JEVPC-RGB-{fed}-fp32', 'TwoHead-RGB-{fed}-bf16',
           'VJEPA-Hybrid-RGB-{fed}-bf16']
PERTS = {'depth': ['clean', 'noise', 'blur', 'occlude', 'dropout', 'scale'],
         'rgb':   ['clean', 'noise', 'blur', 'occlude', 'dropout', 'dim', 'tint']}


def gauss_kernel(sigma, k=7):
    t = torch.arange(k) - k // 2
    g = torch.exp(-t.float() ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def perturb(fr, kind, seed, input_mode):
    """fr: (N, C, 64, 64) in [0, 1]."""
    g = torch.Generator().manual_seed(seed); x = fr.clone(); N, C = x.shape[:2]
    if kind == 'clean':
        return x
    if kind == 'noise':
        return (x + 0.05 * torch.randn(x.shape, generator=g)).clamp(0, 1)
    if kind == 'blur':
        k = gauss_kernel(1.5).repeat(C, 1, 1, 1)
        return F.conv2d(F.pad(x, (3, 3, 3, 3), mode='replicate'), k, groups=C)
    if kind == 'occlude':
        t = torch.randint(0, 48, (N,), generator=g); l = torch.randint(0, 48, (N,), generator=g)
        v = 0.05 if input_mode == 'depth' else 0.5
        for i in range(N):
            x[i, :, t[i]:t[i] + 16, l[i]:l[i] + 16] = v
        return x
    if kind == 'dropout':
        drop = torch.rand((N, 1) + x.shape[2:], generator=g) < 0.10
        return torch.where(drop, torch.ones_like(x), x)
    if kind == 'scale':
        return (x * 0.9).clamp(0, 1)
    if kind == 'dim':
        return (x * 0.7).clamp(0, 1)
    if kind == 'tint':
        x[:, 0] += 0.1; x[:, 2] -= 0.1
        return x.clamp(0, 1)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', choices=['depth', 'rgb'], default='depth')
    ap.add_argument('--recs', choices=['set_b', 'scannet'], default='set_b')
    ap.add_argument('--models', nargs='+', default=DEFAULT)
    ap.add_argument('--compare', nargs=2, action='append', default=[], metavar=('A', 'B'))
    a = ap.parse_args()
    RECS = RECSETS[a.recs]
    stride = seq = 10
    dev = torch.device('cuda')
    perts = PERTS[a.input]
    stems = [m for m in a.models if not m.startswith('Random') and (REGISTRY[m]['ch'] == 3 or a.input == 'depth')]
    data = {r: load_frames(r, a.input) for r in RECS}
    pf = {(r, p): perturb(torch.from_numpy(data[r][0]), p, 1000 * RECS.index(r) + perts.index(p), a.input)
          for r in RECS for p in perts}
    res = {}
    for stem in stems:
        fn = load_model(stem, seq, dev, a.input); ch = REGISTRY[stem]["ch"]; print(f"== {stem}", flush=True)
        for r in RECS:
            lat = {p: extract(fn, ch, pf[(r, p)], stride, seq, dev) for p in perts}
            pos = data[r][1][lat['clean'][1]]
            for h, zc in lat['clean'][0].items():
                if h == ' LC':
                    continue
                name = full_name(stem, 'Depth' if a.input == 'depth' else 'RGB') + h
                rid = np.random.default_rng(7).integers(0, len(zc), (5000, 2)); rid = rid[rid[:, 0] != rid[:, 1]]
                med = np.median(np.linalg.norm(zc[rid[:, 0]] - zc[rid[:, 1]], axis=1))
                for p in perts:
                    zp = lat[p][0][h]
                    pe, mi = score(zp, pos, r, stride)
                    res.setdefault((name, p), []).append((pe, mi, np.median(np.linalg.norm(zp - zc, axis=1)) / med))
        del fn; torch.cuda.empty_cache()
    res = {k: np.array(v) for k, v in res.items()}
    names = list(dict.fromkeys(k[0] for k in res))

    lines = [f"Perturbation test, {a.recs} ({len(RECS)} recordings), input={a.input}, stride 10. Clean column = absolute; others = retention (perturbed/clean)"]
    for j, metric in enumerate(('Pearson (log) retention', 'MI retention', 'latent shift (lower = more stable)')):
        lines.append(f"\n#### {metric} ####")
        lines.append(f"{'output':38s}" + "".join(f"{p:>9s}" for p in perts))
        for n in names:
            c = res[(n, 'clean')][:, j]
            if j < 2:
                vals = [c.mean()] + [(res[(n, p)][:, j] / c).mean() for p in perts[1:]]
            else:
                vals = [res[(n, p)][:, j].mean() for p in perts]
            lines.append(f"{n:38s}" + "".join(f"{v:9.3f}" for v in vals))
    if a.compare:
        lines.append("\n#### paired tests (9 recordings): A vs B, per perturbation ####")
    for x, y in a.compare:
        for p in perts[1:]:
            s = []
            for j, m in enumerate(('Pearson ret', 'MI ret', 'shift')):
                u = res[(x, p)][:, j] / (res[(x, 'clean')][:, j] if j < 2 else 1)
                v = res[(y, p)][:, j] / (res[(y, 'clean')][:, j] if j < 2 else 1)
                better = (u > v).sum() if j < 2 else (u < v).sum()
                s.append(f"{m} {(u - v).mean():+.3f} A better {better}/{len(u)} (p={ttest_rel(u, v).pvalue:.3g})")
            lines.append(f"{p:8s} {x} vs {y}: " + " | ".join(s))
    text = "\n".join(lines); print(text)
    out = ROOT / 'results' / (f'perturb_{a.input}' + ('' if a.recs == 'set_b' else f'_{a.recs}') + '.txt'); out.write_text(text + "\n")
    print(f"\n-> {out}")


if __name__ == '__main__':
    main()
