"""
Training script for JEPA-TTR.

64x64, 128-dim latent, multi-environment continual learning — matches
hpc_depth64.pth / hpc_rgb.pth's resolution/data convention. Supports both
depth (.npy, in_channels=1) and RGB (.jpg/.png, in_channels=3). No decoder:
loss is smooth-L1 between the predictor's output and the EMA target
encoder's latent (stop-gradient), frames shifted by one.

Usage:
    # Depth
    python3 models/hpc_predictive_coder/jepa/train.py \
        --data_paths \
            data/rec_wh_01/depth64 data/rec_wh_02/depth64 data/rec_wh_03/depth64 \
            data/rec_wh_04/depth64 data/rec_wh_05/depth64 data/rec_wh_06/depth64 \
        --save_path weights/jepatr/jepatr.pth \
        --epochs 200 --batch_size 16

    # RGB
    python3 models/hpc_predictive_coder/jepa/train.py \
        --data_paths \
            data/rec_wh_01/images64 data/rec_wh_02/images64 data/rec_wh_03/images64 \
            data/rec_wh_04/images64 data/rec_wh_05/images64 data/rec_wh_06/images64 \
        --in_channels 3 \
        --save_path weights/jepatr_rgb/jepatr.pth \
        --epochs 300 --batch_size 128 --lr 3e-4
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from jepatr import JEPATR


class DepthSequenceDataset(Dataset):
    """Sliding window sequence dataset — supports depth (.npy) and RGB (.jpg/.png)."""

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


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """
    VICReg-style covariance regularization: penalizes off-diagonal covariance
    between channels, decorrelating them so the representation can't collapse
    onto a low-dimensional subspace while still passing the per-channel
    variance check.

    z: (N, C) -- N tokens (flattened batch/time/space), C channels.

    variance_loss alone only guarantees each channel individually has some
    spread -- it says nothing about channels moving together. Confirmed
    empirically: a depth checkpoint that never triggered the diversity-check
    collapse alert (pairwise cosine similarity between different frames
    stayed healthy, 0.83-0.93, the whole training run) had an effective
    rank of only ~2 out of 128 dimensions -- the top 2 eigenvalues of the
    embedding covariance matrix captured 99% of the variance across 500
    frames of a held-out recording. That's a softer collapse the pairwise
    diversity check can't see (two points at different positions along the
    same 2D line still look "different" to cosine similarity), and it
    plausibly explains why even the best depth run massively underperforms
    the VPC baseline on downstream Pearson correlation despite "passing"
    every collapse check we had. This term is what's needed to actually
    spread information across the full channel dimensionality instead of
    a couple of dominant directions.
    """
    N, C = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (N - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return (off_diag ** 2).sum() / C


def temporal_rank_loss(z_ctx: torch.Tensor, deltas=(1, 2, 4, 8), margin: float = 0.1) -> torch.Tensor:
    """
    Ordinal temporal-rank loss (ICML-discussion idea, not from the original
    CPC/JEPA papers): for every anchor position t and every pair of temporal
    offsets Delta_a < Delta_b in `deltas` with t + Delta_b < L, enforce the
    context-encoder's output at t+Delta_a is closer to t's output than
    t+Delta_b's is -- margin-ranking, not exact distance matching, which is
    deliberate: the robot stops, turns, and revisits locations, so "latent
    distance grows linearly with Delta" is false on many individual windows
    even though it's true on average. Only asking for the *ordering* is
    robust to that, and needs no odometry at all -- purely self-supervised
    from sequence order.

    z_ctx: (B, L, C, H, W) -- the raw PER-FRAME context-encoder output at
    every position (JEPATR.forward(..., return_full=True)'s third return
    value), BEFORE the causal predictor mixes frames together -- classic
    slow-feature-analysis style, regularizing the encoder itself to be
    temporally ordered, not the predictor's causally-mixed output.

    deltas=(1,2,4,8): fit to seq_len=10 (every JEPATR run so far) -- the
    screenshot's suggested {1,2,4,8,16} doesn't fit a 10-frame window (16
    alone needs 17 frames), so 16 is dropped rather than padded around.
    margin=0.1: untuned first guess, same status as cov_weight=1.0 when that
    was added -- a sanity-check magnitude, not a swept value.

    Vectors are L2-normalized before distance so the margin is in the same
    [0, 2] range regardless of the encoder's raw output scale (matching
    jepa_loss's existing F.normalize(..., dim=2) convention) -- unnormalized,
    an 8192-dim raw latent's L2 distance could be in the tens to hundreds,
    making a margin of 0.1 meaningless.
    """
    B, L, C, H, W = z_ctx.shape
    flat = F.normalize(z_ctx.reshape(B, L, -1), dim=-1)

    sorted_deltas = sorted(deltas)
    losses = []
    for t in range(L):
        for i, da in enumerate(sorted_deltas):
            for db in sorted_deltas[i + 1:]:
                if t + db >= L:
                    continue
                d_a = (flat[:, t] - flat[:, t + da]).norm(dim=-1)
                d_b = (flat[:, t] - flat[:, t + db]).norm(dim=-1)
                losses.append(F.relu(d_a - d_b + margin))

    if not losses:
        return torch.zeros((), device=z_ctx.device)
    return torch.stack(losses).mean()


def jepa_loss(predicted: torch.Tensor, target: torch.Tensor,
              var_weight: float = 5.0, cov_weight: float = 1.0) -> torch.Tensor:
    """
    Smooth-L1 between L2-normalized latents (each of the 64 spatial patches
    normalized independently, dim=2 is the channel/latent_dim axis of the
    (B, L, C, H, W) tensors) plus variance and covariance regularization
    terms on the predictor's raw (pre-normalization) output.

    The L2-normalization anchors embedding *scale* -- without it, the
    predictor can shrink the loss by growing its output magnitude instead
    of getting the direction right, which fed back into BatchNorm's
    running_var (copied verbatim into the target encoder every EMA step)
    and ran away. Confirmed via a live run: unnormalized loss was stable
    through epoch ~20, then predictor weight norm and max BN running_var
    blew up together (31->82 and 43->3347 between epoch 30 and 36), NaN by
    epoch 38.

    The variance term anchors per-channel *diversity* -- see variance_loss's
    docstring for the collapse this fixes on its own (full collapse to a
    single point, caught by the pairwise diversity check).

    The covariance term anchors cross-channel *independence* -- see
    covariance_loss's docstring for the softer, harder-to-detect collapse
    this fixes (effective rank ~2/128 even while passing the diversity
    check).
    """
    flat = predicted.permute(0, 1, 3, 4, 2).reshape(-1, predicted.shape[2])
    var_reg = variance_loss(flat)
    cov_reg = covariance_loss(flat)

    predicted = F.normalize(predicted, dim=2)
    target = F.normalize(target, dim=2)
    pred_loss = F.smooth_l1_loss(predicted, target)

    return pred_loss + var_weight * var_reg + cov_weight * cov_reg


def train_epoch(model, dataloader, optimizer, device, epoch, env_name,
                 global_step, total_steps, base_momentum, rank_weight, rank_deltas, pred_rank_weight):
    model.train()
    total_loss = 0.0
    total_rank_loss = 0.0
    total_pred_rank_loss = 0.0
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for i, sequences in enumerate(dataloader):
        sequences = sequences.to(device)
        predicted, target, z_ctx, predicted_full = model(sequences, return_full=True)
        rank_loss = temporal_rank_loss(z_ctx, deltas=rank_deltas)
        # Same ordinal temporal-rank loss, now ALSO applied to the predictor's
        # full output (predicted_full == get_latents()'s output) -- the stage
        # the downstream Pearson eval actually reads, which previously only got
        # indirect pressure via covariance_loss on the shifted `predicted`.
        # Gated on pred_rank_weight > 0 -- NOT just scaled by it -- so a
        # pred_rank_weight=0 run (e.g. a timing-comparison baseline) actually
        # skips the computation, rather than computing it and multiplying by
        # zero (which still pays the full forward/backward cost).
        if pred_rank_weight > 0:
            pred_rank_loss = temporal_rank_loss(predicted_full, deltas=rank_deltas)
        else:
            pred_rank_loss = torch.zeros((), device=predicted_full.device)
        loss = jepa_loss(predicted, target) + rank_weight * rank_loss + pred_rank_weight * pred_rank_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        momentum = momentum_schedule(global_step, total_steps, base_momentum)
        model.update_target_encoder(momentum)
        global_step += 1

        total_loss += loss.item()
        total_rank_loss += rank_loss.item()
        total_pred_rank_loss += pred_rank_loss.item()
        print(f"  [{env_name}] Epoch {epoch + 1} batch {i}/{len(dataloader)} "
              f"loss={loss.item():.6f} rank_loss={rank_loss.item():.6f} "
              f"pred_rank_loss={pred_rank_loss.item():.6f} momentum={momentum:.5f}", flush=True)

    return (total_loss / len(dataloader), total_rank_loss / len(dataloader),
            total_pred_rank_loss / len(dataloader), global_step)


def validate(model, dataloader, device, rank_weight, rank_deltas, pred_rank_weight):
    model.eval()
    total_loss = 0.0
    total_rank_loss = 0.0
    total_pred_rank_loss = 0.0
    with torch.no_grad():
        for sequences in dataloader:
            sequences = sequences.to(device)
            predicted, target, z_ctx, predicted_full = model(sequences, return_full=True)
            rank_loss = temporal_rank_loss(z_ctx, deltas=rank_deltas)
            if pred_rank_weight > 0:
                pred_rank_loss = temporal_rank_loss(predicted_full, deltas=rank_deltas)
            else:
                pred_rank_loss = torch.zeros((), device=predicted_full.device)
            loss = jepa_loss(predicted, target) + rank_weight * rank_loss + pred_rank_weight * pred_rank_loss
            total_loss += loss.item()
            total_rank_loss += rank_loss.item()
            total_pred_rank_loss += pred_rank_loss.item()
    return (total_loss / len(dataloader), total_rank_loss / len(dataloader),
            total_pred_rank_loss / len(dataloader))


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True

    model = JEPATR(
        in_channels=args.in_channels,
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

    # Datasets — one per environment (pass depth64/ or images64/ paths directly)
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

    # Adam + CosineAnnealingLR -- reverted back from the SGD+OneCycleLR
    # experiment (weights/jepatr_rank_sgd/): that run showed SGD+OneCycleLR
    # significantly HURTS JEPATR specifically (stride=10 Pearson 0.384->0.331,
    # paired-t p=0.0063), unlike VPC/CPC/AE where it helped or tied. Likely
    # cause: JEPATR's entire training signal flows through an attention-based
    # causal predictor + a BatchNorm-containing EMA target -- architectures
    # built on attention/batch-statistics are known to train poorly with plain
    # SGD+momentum vs Adam's per-parameter adaptive scaling, unlike VPC/AE's
    # mostly-conv, reconstruction-dominated loss. So: Adam is JEPATR's own best
    # optimizer, kept deliberately different from VPC/AE/CPC's SGD -- see
    # project_jepatr.md's optimizer-ablation note for the full justification
    # (this is the controlled experiment that justifies using different final
    # optimizers per model in the paper).
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Full per-epoch loss history, matching vpc/train.py's / cpc/train.py's convention
    # (jepa/train.py didn't have this yet -- added here, kept from the SGD experiment).
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

        train_losses = []
        train_rank_losses = []
        train_pred_rank_losses = []
        for env_name, loader in train_loaders:
            loss, rank_loss, pred_rank_loss, global_step = train_epoch(
                model, loader, optimizer, device, epoch, env_name,
                global_step, total_steps, args.base_momentum,
                args.rank_weight, args.rank_deltas, args.pred_rank_weight
            )
            train_losses.append(loss)
            train_rank_losses.append(rank_loss)
            train_pred_rank_losses.append(pred_rank_loss)
            print(f"  [{env_name}] train loss = {loss:.6f}  (rank_loss={rank_loss:.6f}, pred_rank_loss={pred_rank_loss:.6f})")

        val_losses = []
        val_rank_losses = []
        val_pred_rank_losses = []
        for env_name, loader in val_loaders:
            loss, rank_loss, pred_rank_loss = validate(model, loader, device, args.rank_weight, args.rank_deltas, args.pred_rank_weight)
            val_losses.append(loss)
            val_rank_losses.append(rank_loss)
            val_pred_rank_losses.append(pred_rank_loss)
            print(f"  [{env_name}] val loss   = {loss:.6f}  (rank_loss={rank_loss:.6f}, pred_rank_loss={pred_rank_loss:.6f})")

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        print(f"\n  avg train = {avg_train:.6f} (rank={np.mean(train_rank_losses):.6f}, pred_rank={np.mean(train_pred_rank_losses):.6f})"
              f"  avg val = {avg_val:.6f} (rank={np.mean(val_rank_losses):.6f}, pred_rank={np.mean(val_pred_rank_losses):.6f})")

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
    parser = argparse.ArgumentParser(description="Train JEPATR")

    parser.add_argument('--data_paths', nargs='+', required=True,
                         help='One or more depth64 environment directories')
    parser.add_argument('--save_path', type=str, default='models/hpc_predictive_coder/jepa/jepatr.pth')
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    # Adam + CosineAnnealingLR -- JEPATR's own best optimizer (SGD+OneCycleLR
    # tried and found to significantly HURT it, see main()'s comment above).
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base_momentum', type=float, default=0.996)
    parser.add_argument('--max_depth', type=float, default=20.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=10)

    # Ordinal temporal-rank loss (see temporal_rank_loss() docstring), applied
    # to the raw per-frame encoder z_ctx. rank_weight=0 recovers the old loss.
    parser.add_argument('--rank_weight', type=float, default=1.0)
    parser.add_argument('--rank_deltas', type=int, nargs='+', default=[1, 2, 4, 8])
    # Same rank loss, ALSO applied to the predictor's full output (predicted_full,
    # == get_latents()) -- the stage the downstream Pearson eval actually reads,
    # which previously only got indirect pressure via covariance_loss. New.
    parser.add_argument('--pred_rank_weight', type=float, default=1.0)

    args = parser.parse_args()
    main(args)
