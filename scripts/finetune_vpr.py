#!/usr/bin/env python3
"""Label-free (temporal) fine-tuning of NetVLAD and AnyLoc on our RGB recordings — fair comparison with our models,
which also learn only from frame order (no positions, no labels).

Data   : RGB images64 of rec_wh_01..06 (same frames our models trained on), held in RAM; last 10% of each recording = val.
Triplets (no labels): anchor i; positive j with |i-j| <= POS_FRAMES (~0.5 m of travel); NEG_PER_ANCHOR negatives with
         |i-j| > NEG_FRAMES (~50 m) in the same recording, hardest one used. Revisits make some negatives false — same
         limitation our models face.
Loss   : triplet margin loss (NetVLAD paper, margin 0.1) on L2-normalised descriptors.
Models (last layers only):
  netvlad : VGG16-NetVLAD-Pitts30k; trainable = conv5 block + NetVLAD layer + whitening; conv1-4 frozen.
  anyloc  : DINOv2 ViT-B/14; trainable = blocks 7-8 (they produce hidden_states[9], the layer AnyLoc reads) + a soft-
            assignment NetVLAD layer initialised from AnyLoc's fitted RGB vocabulary (K=32); rest frozen.
Input  : 64 -> 224 bilinear upsample (same as evaluation). bf16 autocast; full resumable checkpoints.
Usage  : python3 scripts/finetune_vpr.py --model netvlad --save results/finetuned/netvlad_ft.pth
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
import pretrained_backbones as pb

RECS = [f'rec_wh_0{i}' for i in range(1, 7)]
POS_FRAMES, NEG_FRAMES, NEG_PER_ANCHOR, MARGIN = 20, 2000, 10, 0.1


def load_rgb(rec):
    d = ROOT / 'data' / rec / 'images64'
    files = sorted(list(d.glob('*.jpg')) + list(d.glob('*.png')))
    return np.stack([np.asarray(Image.open(f), dtype=np.uint8) for f in files])        # (N, 64, 64, 3)


class Triplets:
    def __init__(self, split, seed=0):
        self.data = []
        for r in RECS:
            a = load_rgb(r); n_val = int(len(a) * 0.1)
            self.data.append(a[:-n_val] if split == 'train' else a[-n_val:])
        self.rng = np.random.default_rng(seed)

    def batch(self, B):
        anc, pos, neg = [], [], []
        for _ in range(B):
            r = self.rng.integers(len(self.data)); a = self.data[r]; N = len(a)
            i = self.rng.integers(N)
            j = int(np.clip(i + self.rng.integers(-POS_FRAMES, POS_FRAMES + 1), 0, N - 1))
            far = np.concatenate([np.arange(0, max(0, i - NEG_FRAMES)), np.arange(min(N, i + NEG_FRAMES), N)])
            if len(far) < NEG_PER_ANCHOR:                                               # tiny split: fall back to other recordings
                o = self.data[(r + 1) % len(self.data)]; ks = [(o, k) for k in self.rng.integers(len(o), size=NEG_PER_ANCHOR)]
            else:
                ks = [(a, k) for k in self.rng.choice(far, NEG_PER_ANCHOR, replace=False)]
            anc.append(a[i]); pos.append(a[j]); neg.extend(x[k] for x, k in ks)
        t = lambda L: torch.from_numpy(np.stack(L)).permute(0, 3, 1, 2).float() / 255.0
        return t(anc), t(pos), t(neg)                                                   # (B,3,64,64) x2, (B*M,3,64,64)


# ------------------------------------------------------------------ models
class SoftVLAD(nn.Module):
    def __init__(self, centers, alpha=100.0):
        super().__init__()
        self.centers = nn.Parameter(centers.clone())                                    # (K, D)
        self.alpha = alpha

    def forward(self, t):                                                               # t: (B, N, D) L2-normalised tokens
        s = F.softmax(-self.alpha * torch.cdist(t, self.centers.unsqueeze(0).expand(len(t), -1, -1)) ** 2, dim=-1)   # (B,N,K)
        # sum_n s_nk (t_n - c_k) = (s^T t)_k - (sum_n s_nk) c_k  -- avoids the (B,N,K,D) intermediate (~7 GB)
        resid = torch.einsum('bnk,bnd->bkd', s, t) - s.sum(1).unsqueeze(-1) * self.centers.unsqueeze(0)
        resid = F.normalize(resid, dim=-1)
        return F.normalize(resid.reshape(len(t), -1), dim=1)


def build_trainable(name, dev):
    if name == 'netvlad':
        f = pb.build_netvlad(dev)                                                        # closure; rebuild modules to train
        from scipy.io import loadmat
        from torchvision import models
        mat = loadmat(str(pb.PRE / 'Pitts30K_struct.mat'), struct_as_record=False, squeeze_me=True)
        vgg = list(models.vgg16().children())[0]
        backbone = nn.Sequential(*list(vgg.children())[:-2])
        for layer, ml in zip(backbone.children(), mat['net'].layers):
            if isinstance(layer, nn.Conv2d):
                layer.weight = nn.Parameter(torch.tensor(ml.weights[0]).float().permute(3, 2, 0, 1))
                layer.bias = nn.Parameter(torch.tensor(ml.weights[1]).float())
        vlad = pb.NetVLADLayer()
        vlad.score_proj.weight = nn.Parameter(torch.tensor(mat['net'].layers[30].weights[0]).float().permute(1, 0).unsqueeze(-1))
        vlad.centers = nn.Parameter(-torch.tensor(mat['net'].layers[30].weights[1]).float())
        whiten = nn.Linear(vlad.output_dim, 4096)
        whiten.weight = nn.Parameter(torch.tensor(mat['net'].layers[33].weights[0]).float().squeeze().permute(1, 0))
        whiten.bias = nn.Parameter(torch.tensor(mat['net'].layers[33].weights[1].squeeze()).float())
        mean = torch.tensor(np.array(mat['net'].meta.normalization.averageImage[0, 0], dtype=np.float32)).view(1, 3, 1, 1)
        scale = 255.0 if float(mean.max()) > 1.0 else 1.0
        model = nn.ModuleDict({'backbone': backbone, 'vlad': vlad, 'whiten': whiten}).to(dev)
        conv5_start = 24                                                                 # VGG16 features index of conv5_1
        for i, m in enumerate(model['backbone']):
            for p in m.parameters():
                p.requires_grad = i >= conv5_start
        mean = mean.to(dev)
        def fwd(x):
            h = pb._prep(x, imagenet=False) * scale - mean
            d = model['backbone'](h)
            d = F.normalize(d.reshape(d.shape[0], d.shape[1], -1), dim=1)
            return F.normalize(model['whiten'](model['vlad'](d)), dim=1)
        return model, fwd
    # anyloc
    dino = pb._dinov2(dev)
    centers = torch.from_numpy(np.load(pb.PRE / f'anyloc_vitb_l{pb.ANYLOC_LAYER}_k{pb.ANYLOC_K}_rgb.npy'))
    model = nn.ModuleDict({'dino': dino, 'vlad': SoftVLAD(centers)}).to(dev)
    for n, p in model['dino'].named_parameters():
        p.requires_grad = any(f'encoder.layer.{b}.' in n for b in (7, 8))
    def fwd(x):
        t = model['dino'](pixel_values=pb._prep(x), output_hidden_states=True).hidden_states[pb.ANYLOC_LAYER][:, 1:]
        return model['vlad'](F.normalize(t, dim=-1))
    return model, fwd


def triplet_loss(fwd, anc, pos, neg, M):
    B = len(anc)
    z = fwd(torch.cat([anc, pos, neg]))
    za, zp, zn = z[:B], z[B:2 * B], z[2 * B:].view(B, M, -1)
    dp = ((za - zp) ** 2).sum(-1)
    dn = ((za.unsqueeze(1) - zn) ** 2).sum(-1).min(1).values                             # hardest negative
    return F.relu(dp - dn + MARGIN).mean(), (dp < dn).float().mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['netvlad', 'anyloc'], required=True)
    ap.add_argument('--save', required=True)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=24, help='anchors per step (each with 1 positive + 10 negatives)')
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--val_every', type=int, default=250)
    ap.add_argument('--resume', default=None)
    a = ap.parse_args()
    dev = torch.device('cuda'); torch.backends.cudnn.benchmark = True
    model, fwd = build_trainable(a.model, dev)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"{a.model}: trainable {sum(p.numel() for p in params):,} / total {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.Adam(params, lr=a.lr)
    sched = lambda s: 0.5 * (1 + math.cos(math.pi * s / a.steps))                        # cosine to 0
    start = 0
    save = Path(a.save); save.parent.mkdir(parents=True, exist_ok=True)
    latest = save.with_name(save.stem + '_latest.pth')
    if a.resume and Path(a.resume).exists():
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); start = ck['step']
        print(f"resumed at step {start}", flush=True)
    train, val = Triplets('train', seed=start), Triplets('val', seed=12345)
    vbatches = [val.batch(a.batch) for _ in range(20)]
    best = float('inf') if start == 0 else ck.get('best', float('inf'))
    t0 = time.time()
    for step in range(start, a.steps):
        for g in opt.param_groups:
            g['lr'] = a.lr * sched(step)
        model.train()
        if a.model == 'netvlad':
            for i, m in enumerate(model['backbone']):
                m.train(i >= 24)
        anc, pos, neg = (x.to(dev, non_blocking=True) for x in train.batch(a.batch))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, acc = triplet_loss(fwd, anc, pos, neg, NEG_PER_ANCHOR)
        opt.zero_grad(); loss.float().backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 50 == 0:
            print(f"step {step}/{a.steps} loss={loss.item():.4f} triplet_acc={acc.item():.3f} lr={opt.param_groups[0]['lr']:.2e} "
                  f"{(time.time() - t0) / max(1, step - start + 1):.2f}s/step", flush=True)
        if (step + 1) % a.val_every == 0 or step + 1 == a.steps:
            model.eval(); vl, va = [], []
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                for b in vbatches:
                    l, ac = triplet_loss(fwd, *(x.to(dev) for x in b), NEG_PER_ANCHOR); vl.append(l.item()); va.append(ac.item())
            v = float(np.mean(vl)); print(f"  VAL step {step + 1}: loss={v:.4f} triplet_acc={np.mean(va):.3f}", flush=True)
            out = {'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step + 1, 'best': min(best, v), 'name': a.model}
            torch.save(out, latest)
            if v < best:
                best = v; torch.save(out, save); print(f"  *** best saved {save}", flush=True)
    print(f"Done. Best val loss: {best:.4f}", flush=True)


if __name__ == '__main__':
    main()
