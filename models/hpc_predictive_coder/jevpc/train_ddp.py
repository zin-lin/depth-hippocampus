"""
DDP (multi-GPU) training script for JEVPC, intended for the ENUCC HPC cluster's
2x A100 GPU nodes. Same architecture/loss composition as train.py (the proven
single-GPU trainer used for the paper's main JEVPC result -- left untouched),
plus one addition: --optimizer sgd_onecycle, to test whether JEPATR Run 7's
"SGD significantly hurts this attention+EMA+BatchNorm architecture" finding
(p=0.0063, see project_jepatr.md) actually transfers to JEVPC's *mixed*
objective (reconstruction + EMA + rank), or whether VPC's own reconstruction
branch inside the hybrid changes that dynamic enough for SGD to be viable (or
even better) -- an empirical question, not something to assume by analogy.

Launch with torchrun, not python directly:
    torchrun --standalone --nproc_per_node=2 \
        models/hpc_predictive_coder/jevpc/train_ddp.py \
        --data_paths data/rec_wh_01/depth64 ... \
        --save_path weights/jevpc_onecycle/jevpc.pth \
        --optimizer sgd_onecycle --epochs 200 --batch_size 128

See HPC_module/train_hpc_jevpc_ddp.sh for the SLURM submission wrapper.
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from jevpc import JEVPC

sys.path.insert(0, str(Path(__file__).parent.parent / 'jepa'))
from train import temporal_rank_loss, jepa_loss, momentum_schedule


class DepthSequenceDataset(Dataset):
    """Identical to train.py's dataset -- duplicated per this project's
    existing per-trainer convention."""

    def __init__(self, data_path, seq_len=10, max_depth=20.0, files=None):
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.max_depth = max_depth

        if files is not None:
            self.depth_files = files
        else:
            npy = sorted(self.data_path.glob("*.npy"))
            jpg = sorted(self.data_path.glob("*.jpg")) + sorted(self.data_path.glob("*.png"))
            self.depth_files = npy if npy else sorted(jpg)

        if len(self.depth_files) == 0:
            raise ValueError(f"No .npy/.jpg/.png files found in {data_path}")

        self.is_rgb = self.depth_files[0].suffix in ('.jpg', '.jpeg', '.png')
        self.num_sequences = max(0, len(self.depth_files) - seq_len + 1)
        if self.num_sequences == 0:
            raise ValueError(f"Not enough frames: {len(self.depth_files)} < {seq_len}")

    def __len__(self):
        return self.num_sequences

    def _load(self, f):
        if self.is_rgb:
            img = np.array(Image.open(f)).astype(np.float32) / 255.0
            return img.transpose(2, 0, 1)
        else:
            depth = np.load(f).astype(np.float32)
            if depth.ndim == 2:
                depth = depth[np.newaxis]
            depth = np.where(np.isinf(depth) | np.isnan(depth), self.max_depth, depth)
            depth = np.clip(depth, 0, self.max_depth) / self.max_depth
            return depth

    def __getitem__(self, idx):
        frames = [self._load(f) for f in self.depth_files[idx: idx + self.seq_len]]
        return torch.from_numpy(np.stack(frames, axis=0)).float()


def is_main(rank):
    return rank == 0


def train_epoch(model, dataloader, criterion, optimizer, scheduler, step_per_batch,
                 device, epoch, env_name, global_step, total_steps, base_momentum,
                 recon_weight, jepa_weight, pred_rank_weight, rank_deltas, rank):
    model.train()
    total_loss = total_recon = total_jepa = total_pred_rank = 0.0
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for i, sequences in enumerate(dataloader):
        sequences = sequences.to(device, non_blocking=True)
        reconstructed, predicted, target, z_ctx, predicted_full = model(sequences, return_full=True)

        recon_loss = criterion(reconstructed, sequences[:, 1:])
        jloss = jepa_loss(predicted, target)
        if pred_rank_weight > 0:
            pred_rank_loss = temporal_rank_loss(predicted_full, deltas=rank_deltas)
        else:
            pred_rank_loss = torch.zeros((), device=predicted_full.device)

        loss = recon_weight * recon_loss + jepa_weight * jloss + pred_rank_weight * pred_rank_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        if step_per_batch:
            scheduler.step()

        # model.module -- DDP wraps JEVPC, update_target_encoder/custom methods
        # aren't proxied through the wrapper. Each rank computes the identical
        # momentum value (same global_step, since DistributedSampler gives every
        # rank the same number of batches/epoch) and applies it to its own
        # (gradient-synced, therefore bit-identical) context encoder -- no
        # extra cross-rank communication needed for the EMA update itself.
        momentum = momentum_schedule(global_step, total_steps, base_momentum)
        model.module.update_target_encoder(momentum)
        global_step += 1

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_jepa += jloss.item()
        total_pred_rank += pred_rank_loss.item()
        if is_main(rank):
            print(f"  [{env_name}] Epoch {epoch + 1} batch {i}/{len(dataloader)} "
                  f"loss={loss.item():.6f} recon={recon_loss.item():.6f} "
                  f"jepa={jloss.item():.6f} pred_rank={pred_rank_loss.item():.6f} "
                  f"momentum={momentum:.5f}", flush=True)

    n = len(dataloader)
    return (total_loss / n, total_recon / n, total_jepa / n, total_pred_rank / n, global_step)


def validate(model, dataloader, criterion, device, recon_weight, jepa_weight,
             pred_rank_weight, rank_deltas):
    model.eval()
    total_loss = total_recon = total_jepa = total_pred_rank = 0.0
    with torch.no_grad():
        for sequences in dataloader:
            sequences = sequences.to(device, non_blocking=True)
            reconstructed, predicted, target, z_ctx, predicted_full = model(sequences, return_full=True)
            recon_loss = criterion(reconstructed, sequences[:, 1:])
            jloss = jepa_loss(predicted, target)
            if pred_rank_weight > 0:
                pred_rank_loss = temporal_rank_loss(predicted_full, deltas=rank_deltas)
            else:
                pred_rank_loss = torch.zeros((), device=predicted_full.device)
            loss = recon_weight * recon_loss + jepa_weight * jloss + pred_rank_weight * pred_rank_loss
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_jepa += jloss.item()
            total_pred_rank += pred_rank_loss.item()
    n = len(dataloader)
    return (total_loss / n, total_recon / n, total_jepa / n, total_pred_rank / n)


def main(args):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(backend='nccl', device_id=device)
    torch.backends.cudnn.benchmark = True

    if is_main(rank):
        print(f"DDP world_size={world_size}, device={device}")

    model = JEVPC(in_channels=args.in_channels, latent_dim=args.latent_dim,
                  num_heads=args.num_heads).to(device)

    if is_main(rank):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total:,}  (trainable: {trainable:,})")

    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    ckpt = None
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt.get('global_step', 0)
        best_val_loss = ckpt.get('val_loss', float('inf'))
        if is_main(rank):
            print(f"Resumed from epoch {start_epoch}, val_loss={best_val_loss:.6f}")

    model = DDP(model, device_ids=[local_rank])
    criterion = nn.MSELoss() if args.in_channels == 3 else nn.L1Loss()

    env_paths = [Path(p) for p in args.data_paths]
    train_loaders, val_loaders = [], []
    for p in env_paths:
        all_files = sorted(p.glob("*.npy"))
        if len(all_files) == 0:
            all_files = sorted(sorted(p.glob("*.jpg")) + sorted(p.glob("*.png")))
        if len(all_files) == 0:
            raise ValueError(f"No .npy/.jpg/.png files found in {p}")
        n_val_files = max(args.seq_len, int(len(all_files) * 0.1))
        train_files = all_files[:-n_val_files]
        val_files = all_files[-n_val_files:]
        if is_main(rank):
            print(f"  {p.name}: {len(train_files)} train frames / {len(val_files)} val frames (temporal split)")

        train_ds = DepthSequenceDataset(p, args.seq_len, args.max_depth, files=train_files)
        val_ds = DepthSequenceDataset(p, args.seq_len, args.max_depth, files=val_files)

        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

        train_loaders.append((p.name, DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True), train_sampler))
        val_loaders.append((p.name, DataLoader(val_ds, batch_size=args.batch_size,
                            sampler=val_sampler, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False)))

    total_steps = sum(len(loader) for _, loader, _ in train_loaders) * args.epochs

    if args.optimizer == 'adam':
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        step_per_batch = False
    else:  # sgd_onecycle -- the ablation this script exists for
        optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                               lr=args.lr, momentum=0.9, weight_decay=5e-6)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                   total_steps=total_steps)
        step_per_batch = True

    history_dir = Path(args.save_path).parent
    train_loss_path = history_dir / 'train_loss.npy'
    val_loss_path = history_dir / 'val_loss.npy'
    if is_main(rank):
        history_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and train_loss_path.exists() and val_loss_path.exists():
        history = {'train_loss': np.load(train_loss_path).tolist(),
                   'val_loss': np.load(val_loss_path).tolist()}
    else:
        history = {'train_loss': [], 'val_loss': []}

    if args.resume and ckpt is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if not step_per_batch:
            for _ in range(start_epoch):
                scheduler.step()
        else:
            for _ in range(global_step):
                scheduler.step()

    save_path = Path(args.save_path)
    if is_main(rank):
        save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        for _, _, sampler in train_loaders:
            sampler.set_epoch(epoch)

        if is_main(rank):
            print(f"\n{'=' * 50}\nEpoch {epoch + 1}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.2e}")

        train_losses, train_recon, train_jepa_l, train_pred_rank = [], [], [], []
        for env_name, loader, _ in train_loaders:
            loss, recon, jl, pr, global_step = train_epoch(
                model, loader, criterion, optimizer, scheduler, step_per_batch, device,
                epoch, env_name, global_step, total_steps, args.base_momentum,
                args.recon_weight, args.jepa_weight, args.pred_rank_weight, args.rank_deltas, rank
            )
            train_losses.append(loss); train_recon.append(recon)
            train_jepa_l.append(jl); train_pred_rank.append(pr)

        val_losses, val_recon, val_jepa_l, val_pred_rank = [], [], [], []
        for env_name, loader, in val_loaders:
            loss, recon, jl, pr = validate(model, loader, criterion, device, args.recon_weight,
                                            args.jepa_weight, args.pred_rank_weight, args.rank_deltas)
            val_losses.append(loss); val_recon.append(recon)
            val_jepa_l.append(jl); val_pred_rank.append(pr)

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))

        if not step_per_batch:
            scheduler.step()

        if is_main(rank):
            print(f"  avg train = {avg_train:.6f} (recon={np.mean(train_recon):.6f}, "
                  f"jepa={np.mean(train_jepa_l):.6f}, pred_rank={np.mean(train_pred_rank):.6f})  "
                  f"avg val = {avg_val:.6f} (recon={np.mean(val_recon):.6f}, "
                  f"jepa={np.mean(val_jepa_l):.6f}, pred_rank={np.mean(val_pred_rank):.6f})")

            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)
            np.save(train_loss_path, np.array(history['train_loss']))
            np.save(val_loss_path, np.array(history['val_loss']))

            ckpt_out = {
                'epoch': epoch, 'global_step': global_step,
                'model_state_dict': model.module.state_dict(),  # unwrapped -- loadable without DDP
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train, 'val_loss': avg_val,
                'in_channels': args.in_channels, 'latent_dim': args.latent_dim,
                'num_heads': args.num_heads, 'seq_len': args.seq_len,
                'optimizer_name': args.optimizer,
            }
            torch.save(ckpt_out, str(save_path).replace('.pth', '_latest.pth'))

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(ckpt_out, save_path)
                print(f"  *** New best saved: {save_path}")

            if (epoch + 1) % args.save_every == 0:
                torch.save(ckpt_out, f"{save_path}.epoch_{epoch + 1}")

        dist.barrier()  # keep all ranks in lockstep at epoch boundaries

    if is_main(rank):
        print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train JEVPC (DDP, multi-GPU)")
    parser.add_argument('--data_paths', nargs='+', required=True)
    parser.add_argument('--save_path', type=str, default='weights/jevpc_ddp/jevpc.pth')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128, help='per-GPU batch size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd_onecycle'])
    parser.add_argument('--base_momentum', type=float, default=0.996)
    parser.add_argument('--max_depth', type=float, default=20.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--jepa_weight', type=float, default=1.0)
    parser.add_argument('--pred_rank_weight', type=float, default=1.0)
    parser.add_argument('--rank_deltas', type=int, nargs='+', default=[1, 2, 4, 8])
    args = parser.parse_args()
    main(args)
