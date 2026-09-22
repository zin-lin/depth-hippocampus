"""
Training script for JEPATR.

Depth-only, 64x64, 128-dim latent, multi-environment continual learning —
matches hpc_depth64.pth's resolution/data convention. No decoder: loss is
smooth-L1 between the predictor's output and the EMA target encoder's
latent (stop-gradient), frames shifted by one.

Usage:
    python3 models/hpc_predictive_coder/train_jepa.py \
        --data_paths \
            data/rec_wh_01/depth64 data/rec_wh_02/depth64 data/rec_wh_03/depth64 \
            data/rec_wh_04/depth64 data/rec_wh_05/depth64 data/rec_wh_06/depth64 \
        --save_path weights/jepatr/jepatr.pth \
        --epochs 200 --batch_size 16
"""

import argparse
import math
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from jepa.jepatr import JEPATR


class DepthSequenceDataset(Dataset):
    """Sliding window sequence dataset — depth (.npy) only."""

    def __init__(self, data_path, seq_len=10, max_depth=20.0, files=None):
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.max_depth = max_depth

        self.depth_files = files if files is not None else sorted(self.data_path.glob("*.npy"))
        if len(self.depth_files) == 0:
            raise ValueError(f"No .npy files found in {data_path}")

        self.num_sequences = max(0, len(self.depth_files) - seq_len + 1)
        if self.num_sequences == 0:
            raise ValueError(f"Not enough frames: {len(self.depth_files)} < {seq_len}")

        print(f"  {data_path}: {len(self.depth_files)} frames -> {self.num_sequences} sequences")

    def __len__(self):
        return self.num_sequences

    def _load(self, f):
        depth = np.load(f).astype(np.float32)
        if depth.ndim == 2:
            depth = depth[np.newaxis]
        depth = np.where(np.isinf(depth) | np.isnan(depth), self.max_depth, depth)
        depth = np.clip(depth, 0, self.max_depth) / self.max_depth
        return depth  # (1, H, W)

    def __getitem__(self, idx):
        frames = [self._load(f) for f in self.depth_files[idx: idx + self.seq_len]]
        return torch.from_numpy(np.stack(frames, axis=0)).float()


def momentum_schedule(step: int, total_steps: int, base_momentum: float) -> float:
    """Cosine anneal momentum from base_momentum -> 1.0 over training."""
    progress = step / max(1, total_steps)
    return 1.0 - (1.0 - base_momentum) * (math.cos(math.pi * progress) + 1) / 2


def variance_loss(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """
    VICReg-style variance regularization: hinge-penalizes any channel whose
    std across the batch of tokens falls below target_std.

    z: (N, C) -- N tokens (flattened batch/time/space), C channels.

    This is what actually stops the encoder from collapsing to a constant
    output regardless of input. Confirmed empirically on a live 200-epoch
    run: with only L2-norm + EMA + stop-gradient (no variance pressure),
    the raw context encoder output converged to cosine similarity 1.0000
    between totally different frames -- including frames 5000 apart in the
    same trajectory and frames from an entirely different recording -- even
    though the input pixels differed by 0.17 mean absolute difference. Loss
    looked perfectly converged (flat, train==val) throughout; it was
    collapse, not convergence. EMA/stop-gradient alone weren't enough here
    because the predictor uses per-sample LayerNorm, not the batch-wise
    BatchNorm that gives BYOL's predictor its implicit anti-collapse effect.
    """
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(target_std - std).mean()


def jepa_loss(predicted: torch.Tensor, target: torch.Tensor, var_weight: float = 5.0) -> torch.Tensor:
    """
    Smooth-L1 between L2-normalized latents (each of the 64 spatial patches
    normalized independently, dim=2 is the channel/latent_dim axis of the
    (B, L, C, H, W) tensors) plus a variance-regularization term on the
    predictor's raw (pre-normalization) output.

    The L2-normalization anchors embedding *scale* -- without it, the
    predictor can shrink the loss by growing its output magnitude instead
    of getting the direction right, which fed back into BatchNorm's
    running_var (copied verbatim into the target encoder every EMA step)
    and ran away. Confirmed via a live run: unnormalized loss was stable
    through epoch ~20, then predictor weight norm and max BN running_var
    blew up together (31->82 and 43->3347 between epoch 30 and 36), NaN by
    epoch 38.

    The variance term anchors embedding *diversity* -- see variance_loss's
    docstring for the separate collapse this fixes (same run, different
    failure mode, only caught once loss had *already* looked converged).
    """
    var_reg = variance_loss(predicted.permute(0, 1, 3, 4, 2).reshape(-1, predicted.shape[2]))

    predicted = F.normalize(predicted, dim=2)
    target = F.normalize(target, dim=2)
    pred_loss = F.smooth_l1_loss(predicted, target)

    return pred_loss + var_weight * var_reg


def train_epoch(model, dataloader, optimizer, device, epoch, env_name,
                 global_step, total_steps, base_momentum):
    model.train()
    total_loss = 0.0
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for i, sequences in enumerate(dataloader):
        sequences = sequences.to(device)
        predicted, target = model(sequences)
        loss = jepa_loss(predicted, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        momentum = momentum_schedule(global_step, total_steps, base_momentum)
        model.update_target_encoder(momentum)
        global_step += 1

        total_loss += loss.item()
        print(f"  [{env_name}] Epoch {epoch + 1} batch {i}/{len(dataloader)} "
              f"loss={loss.item():.6f} momentum={momentum:.5f}", flush=True)

    return total_loss / len(dataloader), global_step


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for sequences in dataloader:
            sequences = sequences.to(device)
            predicted, target = model(sequences)
            loss = jepa_loss(predicted, target)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True

    model = JEPATR(
        in_channels=1,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}  (trainable: {trainable:,})")

    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt.get('global_step', 0)
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, val_loss={best_val_loss:.6f}")

    # Datasets — one per environment (pass depth64/ paths directly)
    env_paths = [Path(p) for p in args.data_paths]
    train_loaders = []
    val_loaders = []
    for p in env_paths:
        all_files = sorted(p.glob("*.npy"))
        if len(all_files) == 0:
            raise ValueError(f"No .npy files found in {p}")
        n_val_files = max(args.seq_len, int(len(all_files) * 0.1))
        train_files = all_files[:-n_val_files]
        val_files = all_files[-n_val_files:]
        print(f"  {p.name}: {len(train_files)} train frames / {len(val_files)} val frames (temporal split)")
        train_ds = DepthSequenceDataset(p, args.seq_len, args.max_depth, files=train_files)
        val_ds = DepthSequenceDataset(p, args.seq_len, args.max_depth, files=val_files)
        train_loaders.append((p.name, DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)))
        val_loaders.append((p.name, DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)))

    total_steps = sum(len(loader) for _, loader in train_loaders) * args.epochs

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    if args.resume and Path(args.resume).exists():
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for _ in range(start_epoch):
            scheduler.step()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.2e}")

        train_losses = []
        for env_name, loader in train_loaders:
            loss, global_step = train_epoch(
                model, loader, optimizer, device, epoch, env_name,
                global_step, total_steps, args.base_momentum
            )
            train_losses.append(loss)
            print(f"  [{env_name}] train loss = {loss:.6f}")

        val_losses = []
        for env_name, loader in val_loaders:
            loss = validate(model, loader, device)
            val_losses.append(loss)
            print(f"  [{env_name}] val loss   = {loss:.6f}")

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        print(f"\n  avg train = {avg_train:.6f}  avg val = {avg_val:.6f}")

        scheduler.step()

        ckpt = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train,
            'val_loss': avg_val,
            'in_channels': 1,
            'latent_dim': args.latent_dim,
            'num_heads': args.num_heads,
            'seq_len': args.seq_len,
        }

        torch.save(ckpt, str(save_path).replace('.pth', '_latest.pth'))

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(ckpt, save_path)
            print(f"  *** New best saved: {save_path}")

        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, f"{save_path}.epoch_{epoch + 1}")
            print(f"  Periodic checkpoint saved at epoch {epoch + 1}")

    print(f"\nDone. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train JEPATR")

    parser.add_argument('--data_paths', nargs='+', required=True,
                         help='One or more depth64 environment directories')
    parser.add_argument('--save_path', type=str, default='models/hpc_predictive_coder/jepa/jepatr.pth')
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base_momentum', type=float, default=0.996)
    parser.add_argument('--max_depth', type=float, default=20.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=10)

    args = parser.parse_args()
    main(args)
