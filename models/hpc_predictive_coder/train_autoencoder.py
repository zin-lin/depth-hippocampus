"""
Training script for HPCAutoencoder.

Trains a frame-level autoencoder (no temporal attention) on depth or RGB data.

Usage:
    # Depth 64x64 across multiple recordings
    python3 models/hpc_predictive_coder/train_autoencoder.py \
        --data_paths data/rec_12/depth64 data/rec_13/depth64 data/rec_14/depth64 \
        --in_channels 1 --latent_dim 128 \
        --epochs 200 --batch_size 32 \
        --save_path weights/chk4/hpc_ae_depth64.pth

    # RGB 64x64
    python3 models/hpc_predictive_coder/train_autoencoder.py \
        --data_paths data/rec_12/images64 data/rec_13/images64 \
        --in_channels 3 --latent_dim 128 \
        --epochs 200 --batch_size 32 \
        --save_path weights/chk4/hpc_ae_rgb64.pth

    # Resume
    python3 models/hpc_predictive_coder/train_autoencoder.py \
        --data_paths data/rec_12/depth64 \
        --resume weights/chk4/hpc_ae_depth64_latest.pth \
        --save_path weights/chk4/hpc_ae_depth64.pth
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from autoencoder import HPCAutoencoder


class FrameDataset(Dataset):
    """Single-frame dataset for autoencoder training. Supports depth (.npy) and RGB (.jpg/.png)."""

    def __init__(self, files, max_depth=20.0):
        self.files = files
        self.max_depth = max_depth
        self.is_rgb = files[0].suffix in ('.jpg', '.jpeg', '.png')

    def __len__(self):
        return len(self.files)

    def _load(self, f):
        if self.is_rgb:
            img = np.array(Image.open(f)).astype(np.float32) / 255.0
            return img.transpose(2, 0, 1)  # (3, H, W)
        else:
            d = np.load(f).astype(np.float32)
            if d.ndim == 2:
                d = d[np.newaxis]
            d = np.where(np.isinf(d) | np.isnan(d), self.max_depth, d)
            d = np.clip(d, 0, self.max_depth) / self.max_depth
            return d  # (1, H, W)

    def __getitem__(self, idx):
        return torch.from_numpy(self._load(self.files[idx])).float()


def build_loaders(data_paths, batch_size, num_workers, max_depth, val_frac=0.1):
    all_files = []
    for p in data_paths:
        p = Path(p)
        npy = sorted(p.glob('*.npy'))
        imgs = sorted(p.glob('*.jpg')) + sorted(p.glob('*.png'))
        found = npy if npy else sorted(imgs)
        if not found:
            raise ValueError(f'No .npy/.jpg/.png in {p}')
        print(f'  {p.name}: {len(found)} frames')
        all_files.extend(found)

    n_val = max(1, int(len(all_files) * val_frac))
    train_files = all_files[:-n_val]
    val_files   = all_files[-n_val:]
    print(f'  Total: {len(train_files)} train / {len(val_files)} val frames')

    train_ds = FrameDataset(train_files, max_depth)
    val_ds   = FrameDataset(val_files,   max_depth)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, frames in enumerate(loader):
            frames = frames.to(device)
            recon  = model(frames)
            loss   = criterion(recon, frames)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
            if train and i % 50 == 0:
                print(f'    batch {i}/{len(loader)}  loss={loss.item():.6f}', flush=True)
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', nargs='+', required=True)
    parser.add_argument('--save_path',  type=str,   default='hpc_autoencoder.pth')
    parser.add_argument('--resume',     type=str,   default=None)
    parser.add_argument('--in_channels',type=int,   default=1)
    parser.add_argument('--latent_dim', type=int,   default=128)
    parser.add_argument('--epochs',     type=int,   default=200)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--max_depth',  type=float, default=20.0)
    parser.add_argument('--num_workers',type=int,   default=4)
    parser.add_argument('--save_every', type=int,   default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device:     {device}')
    print(f'Latent dim: {args.latent_dim}')
    print(f'Channels:   {args.in_channels}')

    model = HPCAutoencoder(args.in_channels, args.latent_dim).to(device)
    print(f'Params:     {sum(p.numel() for p in model.parameters()):,}')

    start_epoch  = 0
    best_val     = float('inf')

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('val_loss', float('inf'))
        print(f'Resumed from epoch {start_epoch},  best_val={best_val:.6f}')

    print(f'\nLoading data...')
    train_loader, val_loader = build_loaders(
        args.data_paths, args.batch_size, args.num_workers, args.max_depth
    )

    criterion = nn.L1Loss() if args.in_channels == 1 else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    for _ in range(start_epoch):
        scheduler.step()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]['lr']
        print(f'\n{"="*50}')
        print(f'Epoch {epoch+1}/{args.epochs}  lr={lr:.2e}')

        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        print(f'  train={train_loss:.6f}  val={val_loss:.6f}')
        scheduler.step()

        ckpt = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss':           train_loss,
            'val_loss':             val_loss,
            'in_channels':          args.in_channels,
            'latent_dim':           args.latent_dim,
        }

        latest = str(save_path).replace('.pth', '_latest.pth')
        torch.save(ckpt, latest)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, save_path)
            print(f'  *** Best saved → {save_path}')

        if (epoch + 1) % args.save_every == 0:
            periodic = str(save_path).replace('.pth', f'_ep{epoch+1}.pth')
            torch.save(ckpt, periodic)
            print(f'  Checkpoint → {periodic}')

    print(f'\nDone. Best val loss: {best_val:.6f}')


if __name__ == '__main__':
    main()
