"""
Training script for HPCPredictiveCoder.

512-dim latent, 4-stage encoder, multi-environment continual learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model import create_hpc_predictive_coder


class DepthSequenceDataset(Dataset):
    """Sliding window sequence dataset — supports depth (.npy) and RGB (.jpg/.png)."""

    def __init__(self, data_path, seq_len=10, max_depth=20.0, stride=1, files=None):
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.max_depth = max_depth
        self.stride = stride

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

        print(f"  {data_path}: {len(self.depth_files)} frames → {self.num_sequences} sequences ({'rgb' if self.is_rgb else 'depth'})")

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
        start = idx * self.stride
        frames = [self._load(f) for f in self.depth_files[start : start + self.seq_len]]
        return torch.from_numpy(np.stack(frames, axis=0)).float()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, env_name):
    model.train()
    total_loss = 0.0
    for i, sequences in enumerate(dataloader):
        sequences = sequences.to(device)
        predictions = model(sequences)
        loss = criterion(predictions[:, :-1], sequences[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        print(f"  [{env_name}] Epoch {epoch+1} batch {i}/{len(dataloader)} loss={loss.item():.6f}", flush=True)
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for sequences in dataloader:
            sequences = sequences.to(device)
            predictions = model(sequences)
            loss = criterion(predictions[:, :-1], sequences[:, 1:])
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build model
    model = create_hpc_predictive_coder(
        seq_len=args.seq_len,
        in_channels=args.in_channels,
        out_channels=args.in_channels,
        num_heads=args.num_heads,
        latent_dim=args.latent_dim
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, val_loss={best_val_loss:.6f}")

    # Auto-swap directory name for non-default resolution
    data_paths = args.data_paths
    if args.reso == 64:
        data_paths = [p.replace('/images', '/images64').replace('/depth', '/depth64') for p in data_paths]
        print(f"Resolution 64: using images64/depth64 directories")

    # Datasets — one per environment
    env_paths = [Path(p) for p in data_paths]
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
        val_files   = all_files[-n_val_files:]
        print(f"  {p.name}: {len(train_files)} train frames / {len(val_files)} val frames (temporal split)")
        train_ds = DepthSequenceDataset(p, args.seq_len, args.max_depth, files=train_files)
        val_ds   = DepthSequenceDataset(p, args.seq_len, args.max_depth, files=val_files)
        train_loaders.append((p.name, DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)))
        val_loaders.append((p.name, DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)))

    criterion = nn.MSELoss() if args.in_channels == 3 else nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    save_path = Path(args.save_path)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.2e}")

        # Train on all environments
        train_losses = []
        for env_name, loader in train_loaders:
            loss = train_epoch(model, loader, criterion, optimizer, device, epoch, env_name)
            train_losses.append(loss)
            print(f"  [{env_name}] train loss = {loss:.6f}")

        # Validate on all environments
        val_losses = []
        for env_name, loader in val_loaders:
            loss = validate(model, loader, criterion, device)
            val_losses.append(loss)
            print(f"  [{env_name}] val loss   = {loss:.6f}")

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        print(f"\n  avg train = {avg_train:.6f}  avg val = {avg_val:.6f}")

        scheduler.step()

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train,
            'val_loss': avg_val,
        }

        # Save latest always
        torch.save(ckpt, str(save_path).replace('.pth', '_latest.pth'))

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(ckpt, save_path)
            print(f"  *** New best saved: {save_path}")

        # Periodic save
        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, f"{save_path}.epoch_{epoch+1}")
            print(f"  Periodic checkpoint saved at epoch {epoch+1}")

    print(f"\nDone. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HPCPredictiveCoder")

    parser.add_argument('--data_paths', nargs='+', required=True,
                        help='One or more environment data directories')
    parser.add_argument('--reso', type=int, default=None,
                        help='If 64, auto-swap images→images64 in data_paths')
    parser.add_argument('--save_path', type=str, default='hpc_predictive_coder.pth')
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_depth', type=float, default=20.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=5)

    args = parser.parse_args()
    main(args)
