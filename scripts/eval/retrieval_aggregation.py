#!/usr/bin/env python3
"""Does our loop-closure score suffer from the RETRIEVAL PROTOCOL rather than the latents?
Baselines (NetVLAD, AnyLoc) get a global, position-invariant, L2-normalised descriptor; our models were scored on
the raw flattened 8x8x128 spatial grid with plain L2. Here we give our latents the same treatments:
  raw   : flat 8192, L2 distance (current protocol)
  l2n   : flat 8192, L2-normalised (cosine)
  gap   : mean over the 8x8 grid -> 128-d, L2-normalised (shift-invariant)
  vlad  : the 64 grid cells as local descriptors -> hard-assignment VLAD (K=16), intra + L2 norm (AnyLoc-style).
          Vocabulary fitted on the OTHER recording (cross-fit) so it never sees the evaluated one.
Metrics identical to loop_closure.py (R@1, AP, alias@1, median error) on rec_wh_05/06.
Usage: python3 scripts/eval/retrieval_aggregation.py --input depth --models "VPC-RGB-{fed}-fp32" "TwoHead-RGB-{fed}-bf16"
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WH, get_latents
from loop_closure import sqd, EXCL, REVISIT_M, FAR_M

K = 16


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def cells(flat):
    """flat (N, 8192) laid out as (128, 8, 8) -> local descriptors (N, 64, 128)."""
    return flat.reshape(len(flat), 128, 64).transpose(0, 2, 1)


def vlad(loc, centers):
    d = ((loc[:, :, None, :] - centers[None, None]) ** 2).sum(-1)          # (N, 64, K)
    a = d.argmin(-1)
    out = np.zeros((len(loc), len(centers), loc.shape[-1]), np.float32)
    for k in range(len(centers)):
        m = (a == k)[..., None]
        out[:, k] = ((loc - centers[k]) * m).sum(1)
    out = out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-12)
    return l2n(out.reshape(len(loc), -1))


def metrics(desc_by_rec, pos_by_rec):
    hits, scores, correct, far, errs = [], [], [], [], []
    for r in desc_by_rec:
        lat, P = desc_by_rec[r].astype(np.float64), np.sqrt(sqd(pos_by_rec[r].astype(np.float64)))
        L = np.sqrt(sqd(lat))
        for i in range(EXCL + 1, len(lat)):
            cand = L[i, :i - EXCL]; j = int(cand.argmin()); pe = P[i, j]
            scores.append(-cand[j]); correct.append(pe < REVISIT_M); far.append(pe > FAR_M); errs.append(pe)
            if (P[i, :i - EXCL] < REVISIT_M).any():
                hits.append(pe < REVISIT_M)
    c = np.array(correct)[np.argsort(scores)[::-1]]
    ap = ((np.cumsum(c) / np.arange(1, len(c) + 1)) * c).sum() / max(c.sum(), 1)
    return np.mean(hits), ap, np.mean(far), np.median(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', choices=['depth', 'rgb'], default='depth')
    ap.add_argument('--models', nargs='+', default=['VPC-RGB-{fed}-fp32', 'TwoHead-RGB-{fed}-bf16'])
    a = ap.parse_args()
    lats = get_latents(a.models, WH, a.input, 10)
    lines = [f"Retrieval-protocol test, rec_wh_05/06, input={a.input}, stride 10 (VLAD vocab cross-fitted on the other recording)",
             f"\n{'model':32s}{'variant':>7s}{'R@1':>7s}{'AP':>7s}{'alias@1':>9s}{'medErr_m':>10s}"]
    for name, per in lats.items():
        flat = {r: per[r][0] for r in WH}; pos = {r: per[r][1] for r in WH}
        variants = {'raw': flat,
                    'l2n': {r: l2n(flat[r]) for r in WH},
                    'gap': {r: l2n(cells(flat[r]).mean(1)) for r in WH}}
        vd = {}
        for r, other in ((WH[0], WH[1]), (WH[1], WH[0])):
            loc_o = cells(flat[other]).reshape(-1, 128)
            sub = loc_o[np.random.default_rng(0).choice(len(loc_o), min(100_000, len(loc_o)), replace=False)]
            cen = KMeans(n_clusters=K, n_init=3, random_state=0).fit(sub).cluster_centers_.astype(np.float32)
            vd[r] = vlad(cells(flat[r]), cen)
        variants['vlad'] = vd
        for v, d in variants.items():
            r1, apv, al, me = metrics(d, pos)
            lines.append(f"{name:32s}{v:>7s}{r1:7.3f}{apv:7.3f}{al:9.3f}{me:10.2f}")
    text = "\n".join(lines); print(text)
    out = ROOT / 'results' / f'retrieval_aggregation_{a.input}.txt'; out.write_text(text + "\n"); print(f"\n-> {out}")


if __name__ == '__main__':
    main()
