"""
DDP trainer for JEVPC-TwoHead (MoE Stage 1, see jevpc_twohead.py).

Losses, each on its own head:
  h_MAP: recon_weight * recon(decoder(h_map[:, :-1]), x[:, 1:])
  h_LC:  jepa_weight * jepa_loss(h_lc[:, :-1], z_tgt[:, 1:])
         + rank_weight * temporal_rank_loss(h_lc)
Same recipe as JEVPC otherwise (Adam + cosine by default, --optimizer flag,
EMA momentum schedule, full resumable checkpointing, rank-0-only saving).

Launch: torchrun --standalone --nproc_per_node=2 .../jevpc_twohead/train_ddp.py ...
--batch_size is PER-GPU (128 matches every local JEVPC-family run).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from jevpc_twohead import JEVPCTwoHead

sys.path.insert(0, str(Path(__file__).parent.parent / 'jepa'))
from train import temporal_rank_loss, jepa_loss, momentum_schedule


class SequenceDataset(Dataset):
    def __init__(self, files, seq_len=10, max_depth=20.0):
        self.files, self.seq_len, self.max_depth = files, seq_len, max_depth
        self.is_rgb = files[0].suffix in ('.jpg', '.jpeg', '.png')
        self.n = max(0, len(files) - seq_len + 1)
        if self.n == 0:
            raise ValueError(f"Not enough frames: {len(files)} < {seq_len}")

    def __len__(self):
        return self.n

    def _load(self, f):
        if self.is_rgb:
            return (np.array(Image.open(f)).astype(np.float32) / 255.0).transpose(2, 0, 1)
        d = np.load(f).astype(np.float32)
        if d.ndim == 2:
            d = d[np.newaxis]
        d = np.where(np.isinf(d) | np.isnan(d), self.max_depth, d)
        return np.clip(d, 0, self.max_depth) / self.max_depth

    def __getitem__(self, idx):
        return torch.from_numpy(np.stack([self._load(f) for f in self.files[idx: idx + self.seq_len]])).float()


def corrupt(seq, a):
    """Sensor-style corruption of the context input (per sample, each applied with prob a.aug_p):
    Gaussian noise sigma=a.aug_noise; pixel dropout of a.aug_dropout of pixels set to 1.0 in all channels
    (= max depth when fed depth). Same pixel mask across the clip's channels; independent per frame."""
    B = seq.shape[0]
    x = seq
    if a.aug_noise > 0:
        on = (torch.rand(B, 1, 1, 1, 1, device=seq.device) < a.aug_p).float()
        x = (x + on * a.aug_noise * torch.randn_like(x)).clamp(0, 1)
    if a.aug_dropout > 0:
        on = torch.rand(B, 1, 1, 1, 1, device=seq.device) < a.aug_p
        drop = (torch.rand(seq.shape[0], seq.shape[1], 1, seq.shape[3], seq.shape[4], device=seq.device) < a.aug_dropout) & on
        x = torch.where(drop, torch.ones_like(x), x)
    return x


def compute_losses(model, seq, criterion, a, augment=False):
    x_in = corrupt(seq, a) if augment and (a.aug_noise > 0 or a.aug_dropout > 0) else seq
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=a.bf16):
        # targets stay clean: EMA target encoder gets seq, reconstruction compares to clean seq[:, 1:]
        reconstructed, h_map, h_lc, z_tgt = model(x_in, seq) if x_in is not seq else model(seq)
    # losses in FP32: jepa_loss's variance/covariance terms and the rank margins are precision-sensitive
    reconstructed, h_map, h_lc, z_tgt = reconstructed.float(), h_map.float(), h_lc.float(), z_tgt.float()
    recon = criterion(reconstructed, seq[:, 1:])
    jl = jepa_loss(h_lc[:, :-1], z_tgt[:, 1:])
    rank = temporal_rank_loss(h_lc, deltas=tuple(a.rank_deltas)) if a.rank_weight > 0 else torch.zeros((), device=seq.device)
    loss = a.recon_weight * recon + a.jepa_weight * jl + a.rank_weight * rank
    return loss, recon, jl, rank


def main(a):
    rank_id = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(backend='nccl', device_id=device)
    torch.backends.cudnn.benchmark = True
    main_proc = rank_id == 0

    model = JEVPCTwoHead(in_channels=a.in_channels, latent_dim=a.latent_dim, num_heads=a.num_heads,
                         num_blocks=a.num_blocks, stem_channels=a.stem_channels, num_stages=a.num_stages).to(device)
    if main_proc:
        print(f"DDP world_size={world}  trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    start_epoch, global_step, best_val, ckpt = 0, 0, float('inf'), None
    if a.resume and Path(a.resume).exists():
        ckpt = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch, global_step = ckpt['epoch'] + 1, ckpt.get('global_step', 0)
        best_val = ckpt.get('best_val_loss', ckpt.get('val_loss', float('inf')))
        if main_proc:
            print(f"Resumed from epoch {start_epoch}, best_val={best_val:.6f}")

    model = DDP(model, device_ids=[local_rank])
    criterion = nn.MSELoss() if a.in_channels == 3 else nn.L1Loss()

    train_loaders, val_loaders = [], []
    for p in map(Path, a.data_paths):
        files = sorted(p.glob("*.npy")) or sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")))
        n_val = max(a.seq_len, int(len(files) * 0.1))
        tr, va = SequenceDataset(files[:-n_val], a.seq_len), SequenceDataset(files[-n_val:], a.seq_len)
        ts, vs = DistributedSampler(tr, shuffle=True), DistributedSampler(va, shuffle=False)
        train_loaders.append((p.parent.name, DataLoader(tr, a.batch_size, sampler=ts, num_workers=a.num_workers, pin_memory=True, drop_last=True), ts))
        val_loaders.append(DataLoader(va, a.batch_size, sampler=vs, num_workers=a.num_workers, pin_memory=True))
        if main_proc:
            print(f"  {p}: {len(tr)} train / {len(va)} val sequences")

    total_steps = sum(len(l) for _, l, _ in train_loaders) * a.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    if a.optimizer == 'adam':
        opt = optim.Adam(params, lr=a.lr)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-6)
        per_batch = False
    else:
        opt = optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=5e-6)
        sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total_steps)
        per_batch = True
    if ckpt is not None:
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        for _ in range(global_step if per_batch else start_epoch):
            sched.step()

    save_path = Path(a.save_path)
    hist_dir = save_path.parent
    if main_proc:
        hist_dir.mkdir(parents=True, exist_ok=True)
    hist = {k: (np.load(hist_dir / f'{k}.npy').tolist() if ckpt is not None and (hist_dir / f'{k}.npy').exists() else [])
            for k in ('train_loss', 'val_loss')}

    for epoch in range(start_epoch, a.epochs):
        for _, _, s in train_loaders:
            s.set_epoch(epoch)
        if main_proc:
            print(f"\n{'=' * 50}\nEpoch {epoch + 1}/{a.epochs}  lr={opt.param_groups[0]['lr']:.2e}", flush=True)

        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        model.train()
        tr = np.zeros(4); n = 0
        for env, loader, _ in train_loaders:
            for i, seq in enumerate(loader):
                seq = seq.to(device, non_blocking=True)
                loss, recon, jl, rank = compute_losses(model, seq, criterion, a, augment=True)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                if per_batch:
                    sched.step()
                momentum = momentum_schedule(global_step, total_steps, a.base_momentum)
                model.module.update_target_encoder(momentum)
                global_step += 1
                tr += [loss.item(), recon.item(), jl.item(), rank.item()]; n += 1
                if main_proc and i % 50 == 0:
                    print(f"  [{env}] batch {i}/{len(loader)} loss={loss.item():.6f} recon={recon.item():.6f} "
                          f"jepa={jl.item():.6f} rank={rank.item():.6f} momentum={momentum:.5f}", flush=True)
        tr /= max(n, 1)

        model.eval()
        va = np.zeros(4); n = 0
        with torch.no_grad():
            for loader in val_loaders:
                for seq in loader:
                    seq = seq.to(device, non_blocking=True)
                    va += [x.item() for x in compute_losses(model, seq, criterion, a)]; n += 1
        va /= max(n, 1)

        if not per_batch:
            sched.step()

        if main_proc:
            print(f"  epoch_time={time.time() - t0:.1f}s  peak_vram={torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB  bf16={a.bf16}", flush=True)
            print(f"  avg train = {tr[0]:.6f} (recon={tr[1]:.6f}, jepa={tr[2]:.6f}, rank={tr[3]:.6f})  "
                  f"avg val = {va[0]:.6f} (recon={va[1]:.6f}, jepa={va[2]:.6f}, rank={va[3]:.6f})", flush=True)
            hist['train_loss'].append(float(tr[0])); hist['val_loss'].append(float(va[0]))
            for k, v in hist.items():
                np.save(hist_dir / f'{k}.npy', np.array(v))
            is_best = va[0] < best_val
            best_val = min(best_val, va[0])
            out = {'epoch': epoch, 'global_step': global_step,
                   'model_state_dict': model.module.state_dict(),
                   'optimizer_state_dict': opt.state_dict(),
                   'train_loss': float(tr[0]), 'val_loss': float(va[0]), 'best_val_loss': float(best_val),
                   'in_channels': a.in_channels, 'latent_dim': a.latent_dim, 'num_heads': a.num_heads,
                   'num_blocks': a.num_blocks, 'stem_channels': a.stem_channels, 'num_stages': a.num_stages,
                   'seq_len': a.seq_len, 'optimizer_name': a.optimizer,
                   'aug_noise': a.aug_noise, 'aug_dropout': a.aug_dropout, 'aug_p': a.aug_p}
            torch.save(out, str(save_path).replace('.pth', '_latest.pth'))
            if is_best:
                torch.save(out, save_path)
                print(f"  *** New best saved: {save_path}", flush=True)
            if (epoch + 1) % a.save_every == 0:
                torch.save(out, f"{save_path}.epoch_{epoch + 1}")
        dist.barrier()
        if a.stop_after and epoch + 1 - start_epoch >= a.stop_after:
            break

    if main_proc:
        print(f"\nDone. Best val loss: {best_val:.6f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train JEVPC-TwoHead (DDP)")
    ap.add_argument('--data_paths', nargs='+', required=True)
    ap.add_argument('--save_path', default='weights/jevpc_twohead_rgb/jevpc_twohead.pth')
    ap.add_argument('--resume', default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--in_channels', type=int, default=3)
    ap.add_argument('--latent_dim', type=int, default=128)
    ap.add_argument('--num_heads', type=int, default=4)
    ap.add_argument('--num_blocks', type=int, default=2, help='attention blocks per predictor head (original: 2)')
    ap.add_argument('--stem_channels', type=int, default=64, help='encoder/decoder width (original: 64)')
    ap.add_argument('--num_stages', type=int, default=None, help='2 = /8 grid (width-scaled option C)')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=128, help='per-GPU')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--optimizer', default='adam', choices=['adam', 'sgd_onecycle'])
    ap.add_argument('--base_momentum', type=float, default=0.996)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--save_every', type=int, default=10)
    ap.add_argument('--recon_weight', type=float, default=1.0)
    ap.add_argument('--jepa_weight', type=float, default=1.0)
    ap.add_argument('--rank_weight', type=float, default=1.0)
    ap.add_argument('--rank_deltas', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--stop_after', type=int, default=0,
                    help='benchmark only: stop after N epochs while keeping the --epochs LR schedule')
    ap.add_argument('--aug_noise', type=float, default=0.0, help='train-time Gaussian noise sigma on context input (0 = off)')
    ap.add_argument('--aug_dropout', type=float, default=0.0, help='train-time pixel dropout fraction -> 1.0 (0 = off)')
    ap.add_argument('--aug_p', type=float, default=0.5, help='per-sample probability of each corruption')
    ap.add_argument('--bf16', action='store_true',
                    help='bf16 autocast for forward/backward; losses, params, optimizer and EMA stay FP32')
    main(ap.parse_args())
