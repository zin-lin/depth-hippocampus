"""
Training script for CPC (Contrastive Predictive Coding).

Same 64x64/128-dim/multi-environment convention as jepa/train.py. Loss is
InfoNCE: for each causal context c_t, positives are (a) the real next-frame
latent z_{t+1} (always) and (b) -- only for each window's LAST position -- a
"near-odom" latent: a spatially-near, temporally-excluded revisit of that
frame, when one exists (from models/spatial_index/near_odom.py's
NearOdomIndex, built across all --data_paths recordings). Negatives are every
other (real + near-odom) latent in the minibatch, pooled over spatial dims
first (see cpc_loss() docstring for why).

Motivation for the near-odom positive: the stride=20/window=20 evaluation
showed VPC-Depth specifically collapsing on the perimeter-loop recordings
(perceptual aliasing -- pure depth geometry can't disambiguate one revisit of
a structurally self-similar loop from another) while VPC-RGB held up. CPC's
near-odom term is a direct, explicit countermeasure: an explicit "same place,
different time" training signal that geometry alone doesn't provide.

Usage:
    # Depth
    python3 models/hpc_predictive_coder/cpc/train.py \
        --data_paths \
            data/rec_wh_01/depth64 data/rec_wh_02/depth64 data/rec_wh_03/depth64 \
            data/rec_wh_04/depth64 data/rec_wh_05/depth64 data/rec_wh_06/depth64 \
        --save_path weights/cpc/cpc.pth \
        --epochs 200 --batch_size 128 --lr 3e-4

    # RGB
    python3 models/hpc_predictive_coder/cpc/train.py \
        --data_paths \
            data/rec_wh_01/images64 data/rec_wh_02/images64 data/rec_wh_03/images64 \
            data/rec_wh_04/images64 data/rec_wh_05/images64 data/rec_wh_06/images64 \
        --in_channels 3 \
        --save_path weights/cpc_rgb/cpc.pth \
        --epochs 200 --batch_size 128 --lr 3e-4
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cpc import CPC
from spatial_index.near_odom import NearOdomIndex

sys.path.insert(0, str(Path(__file__).parent.parent / 'jepa'))
from train import temporal_rank_loss


class CPCSequenceDataset(Dataset):
    """
    Sliding-window sequence dataset, extended with an optional near-odom
    auxiliary frame per window (see module docstring). `files` may be a
    train/val slice of the recording's full file list; `file_offset` is that
    slice's starting index in the FULL list, so near-odom lookups (which
    query the index in full-recording frame-index units) stay correct.
    """

    def __init__(self, recording_name, seq_len, max_depth, files,
                 near_odom_index, frame_loader, radius, temporal_exclusion, file_offset=0):
        self.recording_name = recording_name
        self.seq_len = seq_len
        self.max_depth = max_depth
        self.near_odom_index = near_odom_index
        self.frame_loader = frame_loader
        self.radius = radius
        self.temporal_exclusion = temporal_exclusion
        self.file_offset = file_offset

        self.depth_files = files
        if len(self.depth_files) == 0:
            raise ValueError(f"No files for {recording_name}")
        self.is_rgb = self.depth_files[0].suffix in ('.jpg', '.jpeg', '.png')

        self.num_sequences = max(0, len(self.depth_files) - seq_len + 1)
        if self.num_sequences == 0:
            raise ValueError(f"Not enough frames: {len(self.depth_files)} < {seq_len}")

        print(f"  {recording_name}: {len(self.depth_files)} frames -> {self.num_sequences} sequences "
              f"({'rgb' if self.is_rgb else 'depth'})")

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
        window = torch.from_numpy(np.stack(frames, axis=0)).float()

        match = None
        if self.near_odom_index is not None:
            last_frame_global_idx = self.file_offset + idx + self.seq_len - 1
            match = self.near_odom_index.sample(
                self.recording_name, last_frame_global_idx,
                radius=self.radius, temporal_exclusion=self.temporal_exclusion,
            )
        if match is not None:
            aux_file = self.frame_loader[match.recording][match.frame_idx]
            aux = torch.from_numpy(self._load(aux_file)).float()
            aux_valid = True
        else:
            aux = torch.zeros_like(window[0])
            aux_valid = False

        return window, aux, aux_valid


BF16 = False   # set from --bf16 in main(): autocast model forwards only; InfoNCE/rank losses, params, optimizer stay FP32


def _forward(model, window, aux, aux_valid):
    """Model forward (+ near-odom aux encoding) under optional bf16 autocast; returns FP32 tensors."""
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=BF16):
        predicted, target, z = model(window, return_full=True)
        aux_valid_idx = aux_valid.nonzero(as_tuple=True)[0]
        aux_encoded = torch.zeros(window.shape[0], model.latent_dim, predicted.shape[3], predicted.shape[4],
                                  device=window.device)
        if len(aux_valid_idx) > 0:
            aux_encoded[aux_valid_idx] = model.encoder(aux[aux_valid_idx]).float()
    return predicted.float(), target.float(), z.float(), aux_encoded


def cpc_loss(predicted, target, aux_pred, aux_valid, temperature=0.1):
    """
    Multi-positive InfoNCE.

    predicted, target: (B, L-1, C, H, W) -- causal context / real next-latent,
        from CPC.forward(). Pooled over spatial dims (H, W) before scoring:
        flattening every (b, t, h, w) token into one similarity matrix would be
        a B*(L-1)*H*W-square matrix (~155K^2 for batch=128) -- intractable, and
        pools per (h, w) position separately would mean 64 small matrices per
        step for no real benefit on a network this size. Pooling first keeps
        it one (N, N(+M)) matmul, N = B*(L-1).
    aux_pred: (B, C, H, W) -- encoder output (NOT predict_head'd -- it's scored
        as a target/positive, not a prediction) for each window's near-odom
        candidate frame, or garbage where aux_valid is False (filtered out
        below, never used).
    aux_valid: (B,) bool.

    Positives: always the true temporal next-latent (diagonal). Additionally,
    for each window's LAST position, the near-odom latent when aux_valid.
    Negatives: implicit -- every other column in the softmax denominator,
    weighted automatically by the exponential (no separate hard-negative
    mining pass).
    """
    B, Lm1, C, H, W = predicted.shape
    pred_pooled = predicted.mean(dim=[3, 4]).reshape(B * Lm1, C)  # (N, C)
    tgt_pooled = target.mean(dim=[3, 4]).reshape(B * Lm1, C)      # (N, C)
    aux_pooled = aux_pred.mean(dim=[2, 3])                        # (B, C)

    pred_pooled = F.normalize(pred_pooled, dim=1)
    tgt_pooled = F.normalize(tgt_pooled, dim=1)
    aux_pooled = F.normalize(aux_pooled, dim=1)

    N = B * Lm1
    valid_idx = aux_valid.nonzero(as_tuple=True)[0]  # which batch items have a near-odom match
    M = len(valid_idx)

    if M > 0:
        tgt_augmented = torch.cat([tgt_pooled, aux_pooled[valid_idx]], dim=0)  # (N+M, C)
    else:
        tgt_augmented = tgt_pooled

    sim = pred_pooled @ tgt_augmented.T / temperature  # (N, N+M)

    pos_mask = torch.zeros(N, N + M, dtype=torch.bool, device=predicted.device)
    diag = torch.arange(N, device=predicted.device)
    pos_mask[diag, diag] = True

    for m, b in enumerate(valid_idx.tolist()):
        anchor_i = b * Lm1 + (Lm1 - 1)  # last-position anchor for batch item b
        pos_mask[anchor_i, N + m] = True

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -(log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
    return loss.mean(), M


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, env_name, temperature,
                 rank_weight, rank_deltas):
    model.train()
    total_loss = 0.0
    total_rank_loss = 0.0
    total_aux_hits = 0
    total_items = 0

    for i, (window, aux, aux_valid) in enumerate(dataloader):
        window = window.to(device)
        aux = aux.to(device)
        aux_valid = aux_valid.to(device)

        optimizer.zero_grad()

        predicted, target, z, aux_encoded = _forward(model, window, aux, aux_valid)

        cpc_l, n_aux = cpc_loss(predicted, target, aux_encoded, aux_valid, temperature)
        rank_loss = temporal_rank_loss(z, deltas=rank_deltas)
        loss = cpc_l + rank_weight * rank_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()  # OneCycleLR steps every batch, not every epoch -- matches ACCM's trainer.py exactly

        total_loss += loss.item()
        total_rank_loss += rank_loss.item()
        total_aux_hits += n_aux
        total_items += window.shape[0]
        print(f"  [{env_name}] Epoch {epoch + 1} batch {i}/{len(dataloader)} "
              f"loss={loss.item():.6f} rank_loss={rank_loss.item():.6f} "
              f"near_odom_hits={n_aux}/{window.shape[0]}", flush=True)

    return total_loss / len(dataloader), total_rank_loss / len(dataloader), total_aux_hits, total_items


def validate(model, dataloader, device, temperature, rank_weight, rank_deltas):
    model.eval()
    total_loss = 0.0
    total_rank_loss = 0.0
    with torch.no_grad():
        for window, aux, aux_valid in dataloader:
            window = window.to(device)
            aux = aux.to(device)
            aux_valid = aux_valid.to(device)

            predicted, target, z, aux_encoded = _forward(model, window, aux, aux_valid)

            cpc_l, _ = cpc_loss(predicted, target, aux_encoded, aux_valid, temperature)
            rank_loss = temporal_rank_loss(z, deltas=rank_deltas)
            loss = cpc_l + rank_weight * rank_loss
            total_loss += loss.item()
            total_rank_loss += rank_loss.item()
    return total_loss / len(dataloader), total_rank_loss / len(dataloader)


def main(args):
    global BF16
    BF16 = args.bf16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  bf16 autocast: {BF16}")
    torch.backends.cudnn.benchmark = True

    model = CPC(
        in_channels=args.in_channels,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        stem_channels=args.stem_channels,
        num_stages=args.num_stages,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")

    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, val_loss={best_val_loss:.6f}")

    # Full per-epoch loss history, matching ACCM's own Trainer.fit() convention
    # (accm/predictive_coding/trainer.py saves train_loss.npy/val_loss.npy every
    # epoch) so our loss curves are directly comparable/plottable the same way --
    # this is what their Figure 1c-style plot is built from.
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

    env_paths = [Path(p) for p in args.data_paths]
    odom_filename = 'odometry_i.npz' if args.in_channels == 3 else 'odometry.npz'

    frame_loader = {}
    for p in env_paths:
        rec_name = p.parent.name
        all_files = sorted(p.glob("*.npy"))
        if len(all_files) == 0:
            all_files = sorted(p.glob("*.jpg")) + sorted(p.glob("*.png"))
            all_files = sorted(all_files)
        frame_loader[rec_name] = all_files

    # Near-odom positive is optional -- plain CPC (van den Oord et al. 2018)
    # doesn't have this term either, and not every training recording has
    # odometry available (rec_wh_01-06 currently don't). Falls back cleanly
    # to temporal-only InfoNCE when missing, per-recording granularity: any
    # recording without odometry just trains without the near-odom term.
    missing = [p.parent.name for p in env_paths if not (p.parent / odom_filename).exists()]
    if missing:
        print(f"\nNo {odom_filename} for: {missing} -- training as plain (temporal-only) CPC, "
              f"no near-odom positive term. This matches the original CPC formulation exactly.")
        near_odom_index = None
    else:
        print(f"\nBuilding near-odom index across {len(args.data_paths)} recordings "
              f"(radius={args.near_odom_radius}m, temporal_exclusion={args.near_odom_exclusion} frames)")
        recordings = {}
        for p in env_paths:
            rec_name = p.parent.name
            odom = np.load(p.parent / odom_filename)
            recordings[rec_name] = np.stack([odom['x'], odom['y']], axis=1).astype(np.float32)
            assert len(frame_loader[rec_name]) == len(recordings[rec_name]), \
                f"{rec_name}: {len(frame_loader[rec_name])} frames vs {len(recordings[rec_name])} odom entries"
        near_odom_index = NearOdomIndex(recordings)

    train_loaders = []
    val_loaders = []
    for p in env_paths:
        rec_name = p.parent.name
        all_files = frame_loader[rec_name]
        n_val_files = max(args.seq_len, int(len(all_files) * 0.1))
        train_files = all_files[:-n_val_files]
        val_files = all_files[-n_val_files:]
        print(f"  {rec_name}: {len(train_files)} train frames / {len(val_files)} val frames (temporal split)")

        train_ds = CPCSequenceDataset(rec_name, args.seq_len, args.max_depth, train_files,
                                       near_odom_index, frame_loader, args.near_odom_radius,
                                       args.near_odom_exclusion, file_offset=0)
        val_ds = CPCSequenceDataset(rec_name, args.seq_len, args.max_depth, val_files,
                                     near_odom_index, frame_loader, args.near_odom_radius,
                                     args.near_odom_exclusion, file_offset=len(train_files))

        train_loaders.append((rec_name, DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)))
        val_loaders.append((rec_name, DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)))

    # SGD + momentum + weight_decay + OneCycleLR, matching ACCM's own recipe
    # (accm/notebooks/train.ipynb) exactly rather than the Adam+CosineAnnealingLR
    # every other model in this project used -- ACCM's loss curve shape (rises
    # during LR warmup, then falls) is a direct consequence of this schedule,
    # not something Adam+cosine reproduces. OneCycleLR steps every BATCH, not
    # every epoch (see accm/predictive_coding/trainer.py's train() -- scheduler.step()
    # is inside the batch loop, right after optimizer.step()) -- same here.
    steps_per_epoch = sum(len(loader) for _, loader in train_loaders)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)

    if args.resume and Path(args.resume).exists():
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for _ in range(start_epoch * steps_per_epoch):
            scheduler.step()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.2e}")

        train_losses = []
        train_rank_losses = []
        for env_name, loader in train_loaders:
            loss, rank_loss, hits, items = train_epoch(
                model, loader, optimizer, scheduler, device, epoch, env_name, args.temperature,
                args.rank_weight, args.rank_deltas
            )
            train_losses.append(loss)
            train_rank_losses.append(rank_loss)
            print(f"  [{env_name}] train loss = {loss:.6f} (rank={rank_loss:.6f})  "
                  f"near_odom_hit_rate={hits}/{items} ({100 * hits / max(1, items):.1f}%)")

        val_losses = []
        val_rank_losses = []
        for env_name, loader in val_loaders:
            loss, rank_loss = validate(model, loader, device, args.temperature, args.rank_weight, args.rank_deltas)
            val_losses.append(loss)
            val_rank_losses.append(rank_loss)
            print(f"  [{env_name}] val loss   = {loss:.6f} (rank={rank_loss:.6f})")

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        print(f"\n  avg train = {avg_train:.6f} (rank={np.mean(train_rank_losses):.6f})"
              f"  avg val = {avg_val:.6f} (rank={np.mean(val_rank_losses):.6f})")

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        np.save(train_loss_path, np.array(history['train_loss']))
        np.save(val_loss_path, np.array(history['val_loss']))

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train,
            'val_loss': avg_val,
            'in_channels': args.in_channels,
            'latent_dim': args.latent_dim,
            'num_heads': args.num_heads,
            'seq_len': args.seq_len,
            'num_blocks': args.num_blocks,
            'stem_channels': args.stem_channels,
            'num_stages': args.num_stages,
            'bf16': args.bf16,
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
    parser = argparse.ArgumentParser(description="Train CPC")
    parser.add_argument('--data_paths', nargs='+', required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_blocks', type=int, default=2, help='causal attention blocks (option C: 6)')
    parser.add_argument('--stem_channels', type=int, default=64, help='encoder stem width (option C: 192)')
    parser.add_argument('--num_stages', type=int, default=None, help='2 = /8 grid (width-scaled option C)')
    parser.add_argument('--bf16', action='store_true', help='bf16 autocast for model forwards (losses stay FP32)')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--max_depth', type=float, default=20.0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    # SGD + OneCycleLR (max_lr=lr), matching ACCM's own recipe exactly
    # (accm/notebooks/train.ipynb: lr=1e-1, momentum=0.9, weight_decay=5e-6) --
    # not Adam+cosine like the rest of this project's models so far.
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-6)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--near_odom_radius', type=float, default=0.5)
    parser.add_argument('--near_odom_exclusion', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=10)
    # Ordinal temporal-rank loss on the raw per-frame encoder output z (same
    # mechanism/defaults as jepa/train.py's --rank_weight/--rank_deltas --
    # see temporal_rank_loss() there). rank_weight=0 recovers the exact old loss.
    parser.add_argument('--rank_weight', type=float, default=1.0)
    parser.add_argument('--rank_deltas', type=int, nargs='+', default=[1, 2, 4, 8])
    args = parser.parse_args()

    main(args)
