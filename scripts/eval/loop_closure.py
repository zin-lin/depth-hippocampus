#!/usr/bin/env python3
"""Loop-closure (revisit detection) eval. Test A/B is dropped (user: it does nothing for the project).
For each window, find its nearest EARLIER window in latent space, excluding the last 30 windows (~300 frames).
  R@1       : among queries that have a real revisit within 2 m, fraction whose top match is within 2 m
  AP        : average precision over all queries, ranked by match confidence (-latent distance)
  Aliasing@1: fraction of top matches more than 15 m away
  median err: median physical distance of the top match (m)
Latent distances are divided by the median random-pair distance per model per recording (does not change R@1).
Random-init networks are NOT evaluated here (memory: feedback_random_floor_scope). The LC head IS (that's its job).

Examples:
  python3 scripts/eval/loop_closure.py --recs wh --input depth
  python3 scripts/eval/loop_closure.py --recs set_b --input rgb --models "VPC-RGB-{fed}-fp32" "TwoHead-RGB-{fed}-bf16"
Writes results/loop_closure_<recs>_<input>_s<stride>.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SET_B, WH, REGISTRY, get_latents

DEFAULT = ['VPC-Depth-{fed}-fp32', 'VPC-RGB-{fed}-fp32', 'AE-Depth-{fed}-fp32', 'AE-RGB-{fed}-fp32', 'CPC-Depth-{fed}-fp32',
           'JEPATR-Depth-{fed}-fp32', 'JEVPC-Depth-{fed}-fp32', 'JEVPC-RGB-{fed}-fp32', 'TwoHead-RGB-{fed}-fp32',
           'TwoHead-RGB-{fed}-bf16', 'VJEPA-Hybrid-RGB-{fed}-bf16']
EXCL, REVISIT_M, FAR_M = 30, 2.0, 15.0


def sqd(X):
    s = (X ** 2).sum(1)
    return np.clip(s[:, None] + s[None, :] - 2 * X @ X.T, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recs', choices=['wh', 'set_b'], default='wh')
    ap.add_argument('--input', choices=['depth', 'rgb'], default='depth')
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--models', nargs='+', default=DEFAULT)
    ap.add_argument('--allow_random', action='store_true', help='include Random-* floors (off by default: retrieval is not their scope)')
    ap.add_argument('--whiten', type=int, default=0,
                    help='PCA-whiten every descriptor to this many dims (0 = off), fitted on the OTHER wh recording '
                         '(cross-fit, never sees the evaluated one), then L2-normalise. Same treatment for all models.')
    a = ap.parse_args()
    recs = WH if a.recs == 'wh' else SET_B
    stems = [m for m in a.models if (a.allow_random or not m.startswith('Random')) and (REGISTRY[m]['ch'] == 3 or a.input == 'depth')]
    lats = get_latents(stems, recs, a.input, a.stride)
    if a.whiten:
        assert a.recs == 'wh', 'cross-fitted whitening needs the two wh recordings'
        for name, per in lats.items():
            new = {}
            for r, o in ((recs[0], recs[1]), (recs[1], recs[0])):
                X = per[o][0].astype(np.float64); mu = X.mean(0)
                _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
                k = min(a.whiten, len(S))
                W = Vt[:k].T / (S[:k] / np.sqrt(len(X) - 1) + 1e-6)
                Z = (per[r][0] - mu) @ W
                new[r] = ((Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)).astype(np.float32), per[r][1])
            lats[name] = new

    first = next(iter(lats.values()))
    pdist = {r: np.sqrt(sqd(first[r][1].astype(np.float64))) for r in recs}
    elig = {r: sum(1 for i in range(EXCL + 1, len(pdist[r])) if (pdist[r][i, :i - EXCL] < REVISIT_M).any()) for r in recs}
    lines = [f"Loop closure, recs={recs}, input={a.input}, stride={a.stride}, whiten={a.whiten or 'off'}",
             "revisit-eligible queries: " + "  ".join(f"{r}={elig[r]}/{len(pdist[r]) - EXCL - 1}" for r in recs),
             f"\n{'model':38s}{'R@1':>7s}{'AP':>7s}{'alias@1':>9s}{'medErr_m':>10s}"]
    for name, per_rec in lats.items():
        hits, scores, correct, far, errs = [], [], [], [], []
        for r in recs:
            lat, pos = per_rec[r]
            assert len(lat) == len(pdist[r]), (name, r)
            rid = np.random.default_rng(7).integers(0, len(lat), (5000, 2)); rid = rid[rid[:, 0] != rid[:, 1]]
            med = np.median(np.linalg.norm(lat[rid[:, 0]] - lat[rid[:, 1]], axis=1))
            L = np.sqrt(sqd(lat.astype(np.float64))) / med; P = pdist[r]
            for i in range(EXCL + 1, len(lat)):
                cand = L[i, :i - EXCL]; j = int(cand.argmin()); pe = P[i, j]
                scores.append(-cand[j]); correct.append(pe < REVISIT_M); far.append(pe > FAR_M); errs.append(pe)
                if (P[i, :i - EXCL] < REVISIT_M).any():
                    hits.append(pe < REVISIT_M)
        c = np.array(correct)[np.argsort(scores)[::-1]]
        ap_ = ((np.cumsum(c) / np.arange(1, len(c) + 1)) * c).sum() / max(c.sum(), 1)
        lines.append(f"{name:38s}{np.mean(hits):7.3f}{ap_:7.3f}{np.mean(far):9.3f}{np.median(errs):10.2f}")
    lines.append(f"\nchance R@1 (fraction of earlier candidates within {REVISIT_M} m, over eligible queries):")
    for r in recs:
        P = pdist[r]
        fr = [np.mean(P[i, :i - EXCL] < REVISIT_M) for i in range(EXCL + 1, len(P)) if (P[i, :i - EXCL] < REVISIT_M).any()]
        lines.append(f"  {r}: {np.mean(fr):.4f}" if fr else f"  {r}: no revisits")
    text = "\n".join(lines); print(text)
    out = ROOT / 'results' / (f'loop_closure_{a.recs}_{a.input}_s{a.stride}' + (f'_white{a.whiten}' if a.whiten else '') + ('_random' if a.allow_random else '') + '.txt'); out.write_text(text + "\n")
    print(f"\n-> {out}")


if __name__ == '__main__':
    main()
