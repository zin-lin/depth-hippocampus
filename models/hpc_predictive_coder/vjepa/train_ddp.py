"""
DDP trainer for the V-JEPA-style model (see vjepa.py).

Loss: L1(pred, LayerNorm(target)) on MASKED tokens only (pure V-JEPA; no var/cov, no rank).
One multi-block tube mask per batch. EMA momentum schedule 0.996 -> 1.0 and Adam + cosine as in
JEVPC, full resumable checkpointing, rank-0-only saving.

Collapse check (pure L1 with an EMA target has no explicit anti-collapse term): every epoch we log
the std ACROSS SAMPLES of the target and predicted tokens (same position/channel). If the encoder
collapses to a constant, target std -> 0 while the loss also -> 0, so a falling loss alone is not
evidence of learning.

Validation uses a fixed, seeded list of masks so val loss is comparable across epochs.

Launch: torchrun --standalone --nproc_per_node=N .../vjepa/train_ddp.py ...
--batch_size is PER-GPU (128 matches every local JEVPC-family run).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).parent))
from vjepa import VJEPA, sample_tube_mask

sys.path.insert(0, str(Path(__file__).parent.parent / 'jepa'))
from train import momentum_schedule
import math


def vjepa_schedules(step, total, warmup, a):
    """V-JEPA recipe (Bardes et al. 2024): linear LR warmup start_lr -> lr, cosine to final_lr; weight decay
    cosine wd_start -> wd_end; EMA momentum linear base_momentum -> 1.0. All per optimisation step."""
    if step < warmup:
        lr = a.start_lr + (a.lr - a.start_lr) * step / max(1, warmup)
    else:
        prog = (step - warmup) / max(1, total - warmup)
        lr = a.final_lr + 0.5 * (a.lr - a.final_lr) * (1 + math.cos(math.pi * prog))
    prog_all = step / max(1, total)
    wd = a.wd_end + 0.5 * (a.wd_start - a.wd_end) * (1 + math.cos(math.pi * prog_all))
    mom = a.base_momentum + (1.0 - a.base_momentum) * prog_all
    return lr, wd, mom

import importlib.util
# path-based import: this file is also named train_ddp.py, so a plain import would load itself
_spec = importlib.util.spec_from_file_location('twohead_train_ddp', Path(__file__).parent.parent / 'jevpc_twohead' / 'train_ddp.py')
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SequenceDataset = _mod.SequenceDataset


def compute_losses(model, seq, keep, mask, a):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=a.bf16):
        pred, tgt = model(seq, keep, mask)
    pred, tgt = pred.float(), tgt.float()
    loss = F.l1_loss(pred, tgt)
    # std across samples, averaged over positions/channels (collapse indicator, no gradient)
    with torch.no_grad():
        tgt_std, pred_std = tgt.std(0).mean(), pred.std(0).mean()
    return loss, tgt_std, pred_std


def main(a):
    rank_id = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(backend='nccl', device_id=device)
    torch.backends.cudnn.benchmark = True
    main_proc = rank_id == 0

    model = VJEPA(in_channels=a.in_channels, dim=a.dim, depth=a.depth, heads=a.heads, pred_dim=a.pred_dim,
                  pred_depth=a.pred_depth, pred_heads=a.pred_heads, frames=a.seq_len, encoder=a.encoder).to(device)
    if main_proc:
        print(f"DDP world_size={world}  encoder={a.encoder}  trainable params: "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    start_epoch, global_step, best_val, ckpt = 0, 0, float('inf'), None
    if a.resume and Path(a.resume).exists():
        ckpt = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])          # includes target encoder
        start_epoch, global_step = ckpt['epoch'] + 1, ckpt.get('global_step', 0)
        best_val = ckpt.get('best_val_loss', ckpt.get('val_loss', float('inf')))
        if main_proc:
            print(f"Resumed from epoch {start_epoch}, best_val={best_val:.6f}")

    model = DDP(model, device_ids=[local_rank])

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

    grid = model.module.grid
    g_val = torch.Generator().manual_seed(0)
    val_masks = [sample_tube_mask(grid, g_val, device, short_scale=a.short_scale) for _ in range(64)]
    # same mask on every rank for a given step (seeded by step), so DDP ranks stay in lockstep shape-wise
    g_train = torch.Generator()

    total_steps = sum(len(l) for _, l, _ in train_loaders) * a.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    if a.recipe == 'vjepa':
        # AdamW; weight decay only on >=2-D weights (biases, norms, pos/mask tokens excluded), as in V-JEPA
        wd_ok = lambda n, p: p.ndim >= 2 and 'pos' not in n and 'mask_token' not in n
        decay = [p for n, p in model.named_parameters() if p.requires_grad and wd_ok(n, p)]
        no_decay = [p for n, p in model.named_parameters() if p.requires_grad and not wd_ok(n, p)]
        opt = optim.AdamW([{'params': decay, 'weight_decay': a.wd_start}, {'params': no_decay, 'weight_decay': 0.0}], lr=a.start_lr)
        sched = None
        warmup_steps = int(a.warmup_epochs * sum(len(l) for _, l, _ in train_loaders))
    else:
        opt = optim.Adam(params, lr=a.lr)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-6)
    if ckpt is not None:
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        if sched is not None:
            for _ in range(start_epoch):
                sched.step()

    save_path = Path(a.save_path)
    hist_dir = save_path.parent
    if main_proc:
        hist_dir.mkdir(parents=True, exist_ok=True)
    keys = ('train_loss', 'val_loss', 'val_tgt_std', 'val_pred_std')
    hist = {k: (np.load(hist_dir / f'{k}.npy').tolist() if ckpt is not None and (hist_dir / f'{k}.npy').exists() else [])
            for k in keys}

    for epoch in range(start_epoch, a.epochs):
        for _, _, s in train_loaders:
            s.set_epoch(epoch)
        if main_proc:
            print(f"\n{'=' * 50}\nEpoch {epoch + 1}/{a.epochs}  lr={opt.param_groups[0]['lr']:.2e}", flush=True)

        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        model.train()
        tr = np.zeros(3); n = 0; ratio = 0.0
        for env, loader, _ in train_loaders:
            for i, seq in enumerate(loader):
                seq = seq.to(device, non_blocking=True)
                g_train.manual_seed(a.seed * 1_000_003 + global_step)
                keep, mask = sample_tube_mask(grid, g_train, device, short_scale=a.short_scale)
                if a.recipe == 'vjepa':
                    lr_t, wd_t, momentum = vjepa_schedules(global_step, total_steps, warmup_steps, a)
                    opt.param_groups[0]['lr'] = opt.param_groups[1]['lr'] = lr_t
                    opt.param_groups[0]['weight_decay'] = wd_t
                loss, tgt_std, pred_std = compute_losses(model, seq, keep, mask, a)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                if a.recipe != 'vjepa':
                    momentum = momentum_schedule(global_step, total_steps, a.base_momentum)
                model.module.update_target_encoder(momentum)
                global_step += 1
                tr += [loss.item(), tgt_std.item(), pred_std.item()]; n += 1
                ratio += len(mask) / (len(mask) + len(keep))
                if main_proc and i % 50 == 0:
                    print(f"  [{env}] batch {i}/{len(loader)} loss={loss.item():.6f} tgt_std={tgt_std.item():.4f} "
                          f"pred_std={pred_std.item():.4f} masked={len(mask)}/{len(mask) + len(keep)} "
                          f"momentum={momentum:.5f}", flush=True)
        tr /= max(n, 1); ratio /= max(n, 1)

        model.eval()
        va = np.zeros(3); n = 0
        with torch.no_grad():
            for loader in val_loaders:
                for seq in loader:
                    seq = seq.to(device, non_blocking=True)
                    keep, mask = val_masks[n % len(val_masks)]
                    va += [x.item() for x in compute_losses(model, seq, keep, mask, a)]; n += 1
        va /= max(n, 1)
        if sched is not None:
            sched.step()

        if main_proc:
            print(f"  epoch_time={time.time() - t0:.1f}s  peak_vram={torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB  "
                  f"bf16={a.bf16}  mean_mask_ratio={ratio:.2f}", flush=True)
            print(f"  avg train = {tr[0]:.6f} (tgt_std={tr[1]:.4f}, pred_std={tr[2]:.4f})  "
                  f"avg val = {va[0]:.6f} (tgt_std={va[1]:.4f}, pred_std={va[2]:.4f})", flush=True)
            for k, v in zip(keys, (tr[0], va[0], va[1], va[2])):
                hist[k].append(float(v))
            for k, v in hist.items():
                np.save(hist_dir / f'{k}.npy', np.array(v))
            is_best = va[0] < best_val
            best_val = min(best_val, va[0])
            out = {'epoch': epoch, 'global_step': global_step,
                   'model_state_dict': model.module.state_dict(),
                   'optimizer_state_dict': opt.state_dict(),
                   'train_loss': float(tr[0]), 'val_loss': float(va[0]), 'best_val_loss': float(best_val),
                   'in_channels': a.in_channels, 'dim': a.dim, 'depth': a.depth, 'heads': a.heads,
                   'pred_dim': a.pred_dim, 'pred_depth': a.pred_depth, 'pred_heads': a.pred_heads,
                   'encoder': a.encoder, 'seq_len': a.seq_len, 'recipe': a.recipe}
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
    ap = argparse.ArgumentParser(description="Train V-JEPA-style model (DDP)")
    ap.add_argument('--data_paths', nargs='+', required=True)
    ap.add_argument('--save_path', default='weights/vjepa_hybrid_rgb_bf16/vjepa.pth')
    ap.add_argument('--resume', default=None)
    ap.add_argument('--encoder', default='hybrid', choices=['hybrid', 'vit'])
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--in_channels', type=int, default=3)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--depth', type=int, default=None, help='encoder blocks (default: hybrid 3, vit 6)')
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--pred_dim', type=int, default=64)
    ap.add_argument('--pred_depth', type=int, default=4)
    ap.add_argument('--pred_heads', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=128, help='per-GPU')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--base_momentum', type=float, default=0.996)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--recipe', default='ours', choices=['ours', 'vjepa'],
                    help="'ours' = Adam + per-epoch cosine + EMA cosine (JEVPC recipe); 'vjepa' = V-JEPA paper recipe "
                         "(AdamW, LR warmup start_lr->lr then cosine to final_lr, WD cosine wd_start->wd_end, EMA linear base->1)")
    ap.add_argument('--start_lr', type=float, default=2e-4)
    ap.add_argument('--final_lr', type=float, default=1e-6)
    ap.add_argument('--warmup_epochs', type=float, default=27, help='V-JEPA: 40 of 300 epochs -> ~27 of 200')
    ap.add_argument('--wd_start', type=float, default=0.04)
    ap.add_argument('--wd_end', type=float, default=0.4)
    ap.add_argument('--short_scale', type=float, default=0.25, help='area of each of the 8 short mask blocks (V-JEPA: 0.15)')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--save_every', type=int, default=10)
    ap.add_argument('--stop_after', type=int, default=0,
                    help='benchmark only: stop after N epochs while keeping the --epochs LR schedule')
    ap.add_argument('--no_bf16', dest='bf16', action='store_false',
                    help='disable bf16 autocast (default ON; losses, params, optimizer and EMA stay FP32)')
    main(ap.parse_args())
