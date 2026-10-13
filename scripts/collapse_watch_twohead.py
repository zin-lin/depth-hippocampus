#!/usr/bin/env python3
"""Collapse watch for a running TwoHead training job: every time the _latest checkpoint reaches a new
multiple of --every epochs, run BOTH heads (MAP, LC) on a fixed set of clean clips from the held-out
tail of a recording and print one line of health diagnostics per head:
  eff_rank  = exp(entropy of normalised eigenvalues) of the (N, 128) grid-pooled a_L covariance, out of 128
  std       = per-channel std across clips, averaged over channels (-> 0 means every clip gives the same output)
  cos       = mean pairwise cosine similarity between clips' flat a_L (-> 1 means collapse)
Prints 'COLLAPSE ALARM' if eff_rank < 2, std < 1e-3 or cos > 0.95 on either head.
Each printed line is meant to be a Monitor event.

Usage: python3 scripts/collapse_watch_twohead.py --ckpt weights/<run>/jevpc_twohead_latest.pth \
           --frames data/rec_wh_06/images64 [--every 5]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'models/hpc_predictive_coder/jevpc_twohead'))
from jevpc_twohead import JEVPCTwoHead


def load_clips(d, n=256, seq=10):
    files = sorted(list(Path(d).glob('*.jpg')) + list(Path(d).glob('*.png')))
    tail = files[-int(len(files) * 0.1):]                       # held-out 10% (the trainer's val split)
    starts = np.linspace(0, len(tail) - seq, n).astype(int)
    load = lambda f: (np.asarray(Image.open(f), dtype=np.float32) / 255.0).transpose(2, 0, 1)
    return torch.from_numpy(np.stack([np.stack([load(f) for f in tail[s:s + seq]]) for s in starts]))


def eff_rank(x):
    ev = np.clip(np.linalg.eigvalsh(np.cov(x.T)), 0, None); p = ev / ev.sum(); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--every', type=int, default=5)
    ap.add_argument('--poll', type=int, default=30)
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clips = load_clips(a.frames)
    done = set()
    while True:
        try:
            ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
        except Exception:
            time.sleep(a.poll); continue
        ep = ck['epoch'] + 1
        bucket = ep // a.every                     # fire once per crossed block, even if an exact multiple was missed
        if (bucket not in done) and (bucket > 0 or ep == 1):
            done.add(bucket)
            m = JEVPCTwoHead(in_channels=ck['in_channels'], latent_dim=ck['latent_dim'], num_heads=ck['num_heads'],
                             num_blocks=ck.get('num_blocks', 2), stem_channels=ck.get('stem_channels', 64),
                             num_stages=ck.get('num_stages', None))
            m.load_state_dict(ck['model_state_dict']); m = m.to(dev).eval()
            out = []
            with torch.no_grad():
                hm, hl = [], []
                for b in range(0, len(clips), 64):
                    h_map, h_lc = m.get_latents_both(clips[b:b + 64].to(dev))
                    hm.append(h_map[:, -1].float().cpu()); hl.append(h_lc[:, -1].float().cpu())
            alarm = False
            for name, h in (('MAP', torch.cat(hm)), ('LC', torch.cat(hl))):
                pooled = h.mean(dim=(2, 3)).numpy()
                flat = h.reshape(len(h), -1)
                fn = torch.nn.functional.normalize(flat, dim=1)
                cs = fn @ fn.T
                cos = ((cs.sum() - cs.diag().sum()) / (len(cs) * (len(cs) - 1))).item()
                er, sd = eff_rank(pooled), float(pooled.std(0).mean())
                alarm |= er < 2 or sd < 1e-3 or cos > 0.95
                out.append(f"{name} eff_rank={er:6.2f}/128 std={sd:.4f} cos={cos:.3f}")
            print(("COLLAPSE ALARM " if alarm else "") + f"ep{ep}: " + " | ".join(out), flush=True)
            del m; torch.cuda.empty_cache()
        if ep >= 200:
            print("watch finished (epoch 200)", flush=True); return
        time.sleep(a.poll)


if __name__ == '__main__':
    main()
