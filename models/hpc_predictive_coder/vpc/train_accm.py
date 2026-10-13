#!/usr/bin/env python3
"""
Train HPCPredictiveCoder on ACCM's Minecraft village dataset.

Uses ACCM's exact EnvironmentDataset and normalisation so the trained model
is directly comparable to ACCM's released predictive coder weights.
Loss: MSE( predict[:, :-1], input[:, 1:] )  — same as ACCM's trainer.

Usage:
    python3 models/hpc_predictive_coder/train_accm.py \
        --train_dir data/accm/datasets/train-dataset \
        --val_dir   data/accm/datasets/val-dataset \
        --save_dir  weights/accm_rgb
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model import create_hpc_predictive_coder

# ACCM's exact per-channel mean and std (from dataset.py)
ACCM_MEAN = [121.6697 / 255, 149.3242 / 255, 154.9510 / 255]
ACCM_STD  = [40.7521  / 255, 47.7267  / 255, 103.2739 / 255]


class ACCMEnvironmentDataset(Dataset):
    """Replication of ACCM's EnvironmentDataset (dataset.py).

    Each episode folder contains 50 PNG frames (0.png … 49.png) and a
    state.npz with shape (50, 3) for positions.  __getitem__ returns a
    contiguous chunk of seq_len frames from one episode.
    """

    def __init__(self, root: Path, seq_len: int = 20):
        self.episodes = sorted([p for p in Path(root).glob("*") if p.is_dir()])
        self.seq_len = seq_len

        img_exts = ('.png', '.PNG', '.jpg', '.jpeg')
        self._img_files = lambda epi: sorted(
            [f for f in epi.iterdir() if f.suffix in img_exts],
            key=lambda f: f.name
        )
        n_frames = len(self._img_files(self.episodes[0]))
        self.seq_per_epi = n_frames // seq_len

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(ACCM_MEAN, ACCM_STD),
        ])

        print(f"  {root}: {len(self.episodes)} episodes × "
              f"{n_frames} frames → {len(self)} sequences (seq_len={seq_len})")

    def __len__(self):
        return len(self.episodes) * self.seq_per_epi

    def __getitem__(self, idx):
        epi  = self.episodes[idx // self.seq_per_epi]
        fns  = self._img_files(epi)
        off  = (idx % self.seq_per_epi) * self.seq_len

        images = torch.stack(
            [self.transform(Image.open(fns[i])) for i in range(off, off + self.seq_len)],
            dim=0,
        )  # [seq_len, 3, H, W]
        return images


def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total, n = 0.0, 0
    bar = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for images in bar:
        images = images.to(device)
        pred  = model(images)
        loss  = criterion(pred[:, :-1], images[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item()
        n += 1
        bar.set_postfix(loss=f"{loss.item():.5f}")
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, max_batches=50):
    model.eval()
    total, n = 0.0, 0
    for images in loader:
        images = images.to(device)
        pred  = model(images)
        loss  = criterion(pred[:, :-1], images[:, 1:])
        total += loss.item()
        n += 1
        if n >= max_batches:
            break
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir',   type=str, required=True)
    parser.add_argument('--save_dir',  type=str, default='weights/accm_rgb')
    parser.add_argument('--resume',    type=str, default=None)

    parser.add_argument('--seq_len',    type=int,   default=20)
    parser.add_argument('--latent_dim', type=int,   default=128)
    parser.add_argument('--num_heads',  type=int,   default=4)

    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--num_workers',type=int,   default=4)
    parser.add_argument('--save_every', type=int,   default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Datasets — val episodes have 10 frames each (ACCM's pre-packaged paths)
    train_ds = ACCMEnvironmentDataset(args.train_dir, args.seq_len)
    val_ds   = ACCMEnvironmentDataset(args.val_dir,   min(args.seq_len, 10))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model — RGB mode (in_channels=3)
    model = create_hpc_predictive_coder(
        seq_len=args.seq_len,
        in_channels=3,
        out_channels=3,
        num_heads=args.num_heads,
        latent_dim=args.latent_dim,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch   = 0
    best_val_loss = float('inf')

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, val_loss={best_val_loss:.6f}")
        for _ in range(start_epoch):
            scheduler.step()

    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*55}")
        print(f"Epoch {epoch+1}/{args.epochs}   lr={lr:.2e}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, None, device, epoch + 1)
        val_loss   = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  train loss = {train_loss:.6f}   val loss = {val_loss:.6f}")

        ckpt = {
            'epoch':               epoch,
            'model_state_dict':    model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'train_loss':          train_loss,
            'val_loss':            val_loss,
        }

        torch.save(ckpt, save_dir / 'latest.pth')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, save_dir / 'best.pth')
            print(f"  *** New best: {val_loss:.6f}")

        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, save_dir / f'epoch_{epoch+1}.pth')

    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Weights saved to: {save_dir}/")


if __name__ == '__main__':
    main()
