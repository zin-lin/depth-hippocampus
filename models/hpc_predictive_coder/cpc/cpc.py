"""
CPC — Contrastive Predictive Coding (van den Oord et al., 2018, arXiv:1807.03748).

Same causal temporal-attention predictor as JEPATR/HPCPredictiveCoder, but
collapse prevention is *explicit* (InfoNCE contrastive loss against sampled
negatives) rather than *implicit* (EMA target + stop-gradient). Concretely:

  - ONE shared encoder, no EMA/target pair. Unlike JEPATR, there's nothing to
    make asymmetric here — the negative-sampling loss itself is what makes
    the trivial constant-output solution costly (a constant output can't be
    discriminated from negatives, so it's penalized directly), so a frozen
    slow-moving target isn't needed for stability the way it is in JEPA/BYOL.
  - A single-step-ahead (k=1) log-bilinear prediction head W_k (a 1x1 conv,
    applied per spatial patch), following the original CPC paper's
    deliberately simple linear scoring function score(c_t, z_{t+1}) =
    z_{t+1}^T W_k c_t — the point is to put the representation-learning
    burden on the encoder, not on a powerful prediction head. k=1 (not the
    original paper's multi-step k=1..5) to stay a matched-capacity,
    apples-to-apples ablation against JEPATR and the VPC (both one-step).

Pipeline:
  frames (B, L, 1, 64, 64)
    -> encoder (per-frame, shared, grads flow)      -> (B, L, 128, 8, 8)
    -> causal temporal attention (the predictor)     -> (B, L, 128, 8, 8)
    -> predict_head (1x1 conv)                       -> (B, L, 128, 8, 8)
    -> compare predicted[:, :-1] against z[:, 1:] via InfoNCE (cpc/train.py)

Unlike JEPATR's forward(), the "target" returned here is NOT stop-gradient —
CPC's encoder gets gradients from both the context branch (through the
predictor) and the target branch (through the InfoNCE positive/negative
scores) every step, since there's only one encoder.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

try:
    from ..encoder_decoder import HPCEncoder, FFN
    from ..attention import MultiHeadAttention, PositionalEncoding
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from encoder_decoder import HPCEncoder, FFN
    from attention import MultiHeadAttention, PositionalEncoding


class CPC(nn.Module):
    """
    Single shared encoder + causal temporal-attention predictor + linear
    (1x1 conv) prediction head, for InfoNCE-based contrastive predictive
    coding.

    Args:
        in_channels: Input channels (1 for depth, 3 for RGB)
        layers:      Residual blocks per encoder stage [2, 2, 2, 2]
        latent_dim:  Latent channel dim (128 -> 8x8 patches at 64x64 input,
                     matching JEPATR/hpc_depth64.pth for a fair comparison)
        num_heads:   Attention heads for the predictor
        dropout:     Positional encoding dropout (position instantiated but
                     never applied — matches JEPATR/HPCPredictiveCoder exactly,
                     same predictor capacity across all three models)

    Input:  (B, L, C, 64, 64)
    """

    def __init__(
        self,
        in_channels: int = 1,
        layers: List[int] = [2, 2, 2, 2],
        latent_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
        num_blocks: int = 2,
        stem_channels: int = 64,
        num_stages: int = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = HPCEncoder(in_channels, layers, latent_dim, stem_channels, num_stages)

        # Optional extra causal blocks (num_blocks > 2), same pattern as VPC/TwoHead: separate lists so the
        # default num_blocks=2 keeps the original state_dict keys.
        assert num_blocks >= 2, 'num_blocks must be >= 2'
        self.extra_attention = nn.ModuleList([MultiHeadAttention(latent_dim, num_heads) for _ in range(num_blocks - 2)])
        self.extra_ffn = nn.ModuleList([FFN(latent_dim, latent_dim) for _ in range(num_blocks - 2)])

        self.attention_1 = MultiHeadAttention(latent_dim, num_heads)
        self.ffn_1 = FFN(latent_dim, latent_dim)

        self.attention_2 = MultiHeadAttention(latent_dim, num_heads)
        self.ffn_2 = FFN(latent_dim, latent_dim)

        # Instantiated, never called — matches JEPATR/HPCPredictiveCoder's
        # dead positional encoding, so predictor capacity is identical
        # across all three models in the comparison.
        self.position = PositionalEncoding(latent_dim, dropout)

        # W_k, the log-bilinear prediction head: score(c_t, z_{t+1}) =
        # z_{t+1}^T W_k c_t. A 1x1 conv is a per-spatial-patch linear map,
        # matching the 8x8-grid-of-patches latent layout.
        self.predict_head = nn.Conv2d(latent_dim, latent_dim, kernel_size=1, bias=False)

    def _predict(self, f: Tensor) -> Tensor:
        """
        Causal self-attention over context latents — the predictor.
        f: (B, L, C, H, W) -> (B, L, C, H, W), position i only attends to <= i.
        Identical block to JEPATR._predict / HPCPredictiveCoder._apply_temporal_attention.
        """
        L = f.shape[1]
        mask = torch.triu(torch.ones(L, L, device=f.device), diagonal=1).bool()

        residual = f
        f, _ = self.attention_1(f, f, f, mask)
        f = f + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f = self.ffn_1(f) + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f, _ = self.attention_2(f, f, f, mask)
        f = f + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f = self.ffn_2(f) + residual
        f = F.layer_norm(f, f.shape[1:])

        for attn, ffn in zip(self.extra_attention, self.extra_ffn):   # optional extra blocks (num_blocks > 2)
            residual = f
            f, _ = attn(f, f, f, mask)
            f = F.layer_norm(f + residual, f.shape[1:])
            f = F.layer_norm(ffn(f) + f, f.shape[1:])

        return f

    def _encode_sequence(self, x: Tensor) -> Tensor:
        """x: (B, L, C, H, W) -> (B, L, latent_dim, h, w), gradients flow."""
        B, L, C, H, W = x.shape
        z = self.encoder(x.reshape(B * L, C, H, W))
        return z.reshape(B, L, z.shape[1], z.shape[2], z.shape[3])

    def _apply_predict_head(self, c: Tensor) -> Tensor:
        """c: (B, L, C, H, W) -> (B, L, C, H, W), W_k applied per frame/patch."""
        B, L, C, H, W = c.shape
        out = self.predict_head(c.reshape(B * L, C, H, W))
        return out.reshape(B, L, C, H, W)

    def forward(self, x: Tensor, return_full: bool = False):
        """
        Args:
            x: (B, L, C, 64, 64)
            return_full: when True, also return the raw per-frame encoder
                output `z` (B, L, latent_dim, 8, 8) for every position —
                the CPC analog of JEPATR.forward()'s `z_ctx`, for the
                temporal-rank loss (see jepa/train.py's temporal_rank_loss).
        Returns:
            predicted: (B, L-1, latent_dim, 8, 8) — W_k @ (causal context at
                       position i), trained to score highly against frame
                       i+1's real latent and low against negatives
            target:    (B, L-1, latent_dim, 8, 8) — real encoder latents for
                       frames 2..L, gradients flow (no stop-gradient, unlike
                       JEPATR — one shared encoder gets gradients from both
                       branches)
            z (only if return_full): (B, L, latent_dim, 8, 8) — raw per-frame
                       encoder output, before the causal predictor mixes frames
        """
        z = self._encode_sequence(x)
        c = self._predict(z)
        predicted = self._apply_predict_head(c)

        if return_full:
            return predicted[:, :-1], z[:, 1:], z
        return predicted[:, :-1], z[:, 1:]

    def get_latents(self, x: Tensor) -> Tensor:
        """
        Predictor output (before predict_head) for every position, using full
        causal context — the CPC analog of JEPATR.get_latents() /
        HPCPredictiveCoder.get_latents(): the temporally-integrated
        representation to feed into extract_latents.py-style spatial/Pearson
        analysis.

        Args:
            x: (B, L, C, 64, 64)
        Returns:
            (B, L, latent_dim, 8, 8)
        """
        z = self._encode_sequence(x)
        return self._predict(z)


if __name__ == "__main__":
    print("Testing CPC...")

    model = CPC(in_channels=1, latent_dim=128, num_heads=4)
    x = torch.randn(2, 10, 1, 64, 64)

    predicted, target = model(x)
    print(f"Input:      {tuple(x.shape)}")
    print(f"Predicted:  {tuple(predicted.shape)}  requires_grad={predicted.requires_grad}")
    print(f"Target:     {tuple(target.shape)}  requires_grad={target.requires_grad}")

    latents = model.get_latents(x)
    print(f"get_latents: {tuple(latents.shape)}")

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {total:,} (single encoder, no frozen target copy — should be roughly half JEPATR's total)")

    # --- gradient flow sanity check ---
    print("\nTesting gradient flow...")
    # Minimal single-pair InfoNCE-style probe: pull predicted[0,0] toward
    # target[0,0], push away from target[1,0] (a different batch item at the
    # same timestep) -- just to confirm both branches receive real gradients
    # via a contrastive-shaped signal, not the actual training loss.
    pos = (predicted[0, 0] * target[0, 0]).sum()
    neg = (predicted[0, 0] * target[1, 0]).sum()
    loss = -pos + neg
    loss.backward()

    enc_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    pred_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.attention_1.parameters())
    head_has_grad = model.predict_head.weight.grad is not None and model.predict_head.weight.grad.abs().sum() > 0

    print(f"loss: {loss.item():.6f}")
    print(f"encoder received gradients (via both branches): {enc_has_grad} (expected True)")
    print(f"predictor (attention_1) received gradients: {pred_has_grad} (expected True)")
    print(f"predict_head received gradients: {head_has_grad} (expected True)")
