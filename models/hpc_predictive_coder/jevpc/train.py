"""
Training script for JEVPC (Joint-Embedding Visual Predictive Coder).

Combined loss: L1/MSE pixel reconstruction (VPC's loss) + jepa_loss
(normalized smooth-L1 + variance + covariance regularization, JEPATR's loss)
+ predictor-level ordinal temporal-rank loss (JEPATR Run 8's proven win,
applied to the predictor's output only -- the encoder-level "conv aux"
rank loss is dropped since reconstruction now anchors the encoder directly).
See jevpc.py's module docstring for the full architectural motivation.

Optimizer: Adam + CosineAnnealingLR, NOT SGD+OneCycleLR -- JEVPC keeps the
exact same attention-predictor + EMA-target + BatchNorm architecture that
JEPATR Run 7 showed is significantly hurt by SGD (p=0.0063, see
project_jepatr.md). VPC itself prefers SGD (matches ACCM's recipe), but
JEVPC's dominant optimizer-sensitive component (the predictor/EMA machinery)
is unchanged from JEPATR, so Adam is used here too, consistent with that
finding rather than assumed from VPC's recipe.

Checkpointing: saves model_state_dict, optimizer_state_dict, epoch,
global_step (EMA momentum schedule position), train/val loss, and full
per-epoch loss history every single epoch to <save_path>_latest.pth --
same pattern validated on jepatr_pred_rank's mid-run Xid-8 crash (exact LR/
Adam-moment/schedule continuity confirmed across that resume). This run can
hit the same Blackwell GSP watchdog bug at any point; --resume must recover
cleanly with zero lost state beyond the current epoch's partial progress.

Usage:
    python3 models/hpc_predictive_coder/jevpc/train.py \
        --data_paths \
            data/rec_wh_01/depth64 data/rec_wh_02/depth64 data/rec_wh_03/depth64 \
            data/rec_wh_04/depth64 data/rec_wh_05/depth64 data/rec_wh_06/depth64 \
        --save_path weights/jevpc/jevpc.pth \
        --epochs 200 --batch_size 128
"""

import argparse
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from jevpc import JEVPC

sys.path.insert(0, str(Path(__file__).parent.parent / 'jepa'))
from train import temporal_rank_loss, jepa_loss, momentum_schedule


class DepthSequenceDataset(Dataset):
    """Sliding window sequence dataset -- supports depth (.npy) and RGB (.jpg/.png).
    Identical to jepa/train.py's dataset, duplicated per this project's existing
    per-trainer convention (vpc/cpc/jepa/autoencoder each keep their own copy)."""

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

        print(f"  {data_path}: {len(self.depth_files)} frames -> {self.num_sequences} sequences ({'rgb' if self.is_rgb else 'depth'})")

    def __len__(self):
        return self.num_sequences

    def _load(self, f):
        if self.is_rgb:
            img = np.array(Image.open(f)).astype(np.float32) / 255.0  # (H, W, 3)
            return img.transpose(2, 0, 1)  # (3, H, W)
        else:
            depth = np.load(f).astype(np.float32)
            if depth.ndim == 2:
                depth = depth[np.newaxis]
            depth = np.where(np.isinf(depth) | np.isnan(depth), self.max_depth, depth)
            depth = np.clip(depth, 0, self.max_depth) / self.max_depth
            return depth  # (1, H, W)

    def __getitem__(self, idx):
        frames = [self._load(f) for f in self.depth_files[idx: idx + self.seq_len]]
        return torch.from_numpy(np.stack(frames, axis=0)).float()


BF16 = False  # set from --bf16 in main()


def _forward(model, sequences):
    """bf16 autocast over the forward pass (if enabled); outputs cast back to
    FP32 so jepa_loss's variance/covariance terms and the rank margins are
    computed at full precision. Params/optimizer/EMA stay FP32."""
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=BF16):
        outs = model(sequences, return_full=True)
    return tuple(o.float() for o in outs)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, env_name,
                 global_step, total_steps, base_momentum, recon_weight, jepa_weight,
                 pred_rank_weight, rank_deltas):
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_jepa_loss = 0.0
    total_pred_rank_loss = 0.0
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for i, sequences in enumerate(dataloader):
        sequences = sequences.to(device)
        reconstructed, predicted, target, z_ctx, predicted_full = _forward(model, sequences)

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

        momentum = momentum_schedule(global_step, total_steps, base_momentum)
        model.update_target_encoder(momentum)
        global_step += 1

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_jepa_loss += jloss.item()
        total_pred_rank_loss += pred_rank_loss.item()
        print(f"  [{env_name}] Epoch {epoch + 1} batch {i}/{len(dataloader)} "
              f"loss={loss.item():.6f} recon={recon_loss.item():.6f} "
              f"jepa={jloss.item():.6f} pred_rank={pred_rank_loss.item():.6f} "
              f"momentum={momentum:.5f}", flush=True)

    n = len(dataloader)
    return (total_loss / n, total_recon_loss / n, total_jepa_loss / n,
            total_pred_rank_loss / n, global_step)


def validate(model, dataloader, criterion, device, recon_weight, jepa_weight,
             pred_rank_weight, rank_deltas):
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_jepa_loss = 0.0
    total_pred_rank_loss = 0.0
    with torch.no_grad():
        for sequences in dataloader:
            sequences = sequences.to(device)
            reconstructed, predicted, target, z_ctx, predicted_full = _forward(model, sequences)

            recon_loss = criterion(reconstructed, sequences[:, 1:])
            jloss = jepa_loss(predicted, target)
            if pred_rank_weight > 0:
                pred_rank_loss = temporal_rank_loss(predicted_full, deltas=rank_deltas)
            else:
                pred_rank_loss = torch.zeros((), device=predicted_full.device)

            loss = recon_weight * recon_loss + jepa_weight * jloss + pred_rank_weight * pred_rank_loss
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_jepa_loss += jloss.item()
            total_pred_rank_loss += pred_rank_loss.item()

    n = len(dataloader)
    return (total_loss / n, total_recon_loss / n, total_jepa_loss / n,
            total_pred_rank_loss / n)


def main(args):
    global BF16
    BF16 = args.bf16
    print(f"bf16 autocast: {BF16}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True

    model = JEVPC(
        in_channels=args.in_channels,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}  (trainable: {trainable:,})")

    criterion = nn.MSELoss() if args.in_channels == 3 else nn.L1Loss()

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
        print(f"Resumed from epoch {start_epoch}, val_loss={best_val_loss:.6f}")

    env_paths = [Path(p) for p in args.data_paths]
    train_loaders = []
    val_loaders = []
    for p in env_paths:
        all_files = sorted(p.glob("*.npy"))
        if len(all_files) == 0:
            all_files = sorted(p.glob("*.jpg")) + sorted(p.glob("*.png"))
            all_files = sorted(all_files)
        if len(all_files) == 0:
            raise ValueError(f"No .npy/.jpg/.png files found in {p}")
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

    # Adam + CosineAnnealingLR -- see module docstring for why, not SGD+OneCycleLR.
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history_dir = Path(args.save_path).parent
    history_dir.mkdir(parents=True, exist_ok=True)
    train_loss_path = history_dir / 'train_loss.npy'
    val_loss_path = history_dir / 'val_loss.npy'
    if args.resume and train_loss_path.exists() and val_loss_path.exists():
        history = {
            'train_loss': np.load(train_loss_path).tolist(),
            'val_loss': np.load(val_loss_path).tolist(),
        }
        print(f"Resumed loss history: {len(history['train_loss'])} epochs")
    else:
        history = {'train_loss': [], 'val_loss': []}

    if args.resume and Path(args.resume).exists():
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for _ in range(start_epoch):
            scheduler.step()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.2e}")

        train_losses, train_recon, train_jepa_l, train_pred_rank = [], [], [], []
        for env_name, loader in train_loaders:
            loss, recon, jl, pr, global_step = train_epoch(
                model, loader, criterion, optimizer, device, epoch, env_name,
                global_step, total_steps, args.base_momentum,
                args.recon_weight, args.jepa_weight, args.pred_rank_weight, args.rank_deltas
            )
            train_losses.append(loss)
            train_recon.append(recon)
            train_jepa_l.append(jl)
            train_pred_rank.append(pr)
            print(f"  [{env_name}] train loss = {loss:.6f}  (recon={recon:.6f}, jepa={jl:.6f}, pred_rank={pr:.6f})")

        val_losses, val_recon, val_jepa_l, val_pred_rank = [], [], [], []
        for env_name, loader in val_loaders:
            loss, recon, jl, pr = validate(
                model, loader, criterion, device,
                args.recon_weight, args.jepa_weight, args.pred_rank_weight, args.rank_deltas
            )
            val_losses.append(loss)
            val_recon.append(recon)
            val_jepa_l.append(jl)
            val_pred_rank.append(pr)
            print(f"  [{env_name}] val loss   = {loss:.6f}  (recon={recon:.6f}, jepa={jl:.6f}, pred_rank={pr:.6f})")

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        print(f"\n  avg train = {avg_train:.6f} (recon={np.mean(train_recon):.6f}, jepa={np.mean(train_jepa_l):.6f}, pred_rank={np.mean(train_pred_rank):.6f})"
              f"  avg val = {avg_val:.6f} (recon={np.mean(val_recon):.6f}, jepa={np.mean(val_jepa_l):.6f}, pred_rank={np.mean(val_pred_rank):.6f})")

        scheduler.step()

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        np.save(train_loss_path, np.array(history['train_loss']))
        np.save(val_loss_path, np.array(history['val_loss']))

        ckpt = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train,
            'val_loss': avg_val,
            'in_channels': args.in_channels,
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
    parser = argparse.ArgumentParser(description="Train JEVPC")

    parser.add_argument('--data_paths', nargs='+', required=True)
    parser.add_argument('--save_path', type=str, default='weights/jevpc/jevpc.pth')
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    # Adam + CosineAnnealingLR -- see module docstring for why (JEVPC keeps
    # JEPATR's attention+EMA+BatchNorm machinery, which Run 7 showed SGD hurts).
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base_momentum', type=float, default=0.996)
    parser.add_argument('--max_depth', type=float, default=20.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=10)

    # Loss weights. recon_weight=1.0 matches VPC's unweighted L1/MSE directly.
    # jepa_weight=1.0 matches JEPATR's unweighted jepa_loss (itself already
    # internally weighted: smooth_l1 + 5.0*variance + 1.0*covariance).
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--jepa_weight', type=float, default=1.0)
    # Predictor-level ordinal temporal-rank loss only (JEPATR Run 8) -- no
    # encoder-level ("conv aux") rank loss, since reconstruction now anchors
    # the encoder directly.
    parser.add_argument('--pred_rank_weight', type=float, default=1.0)
    parser.add_argument('--rank_deltas', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--bf16', action='store_true',
                        help='bf16 autocast for forward/backward; losses, params, optimizer and EMA stay FP32')

    args = parser.parse_args()
    main(args)
