"""
JEPATR — JEPA with a causal Temporal attention pRedictor.

Structurally, this is Contrastive Predictive Coding (van den Oord et al. 2018):
an autoregressive model (here, causal temporal self-attention, same recipe as
HPCPredictiveCoder's _apply_temporal_attention) consumes past context and
predicts a future latent representation. CPC prevents representational
collapse *explicitly*, via InfoNCE — a contrastive softmax over sampled
negative latents. JEPATR prevents it *implicitly* instead: the target branch
is an EMA (stop-gradient) copy of the context encoder, and that student/teacher
asymmetry is what stops the trivial constant-output solution, with no negative
sampling and no contrastive loss term anywhere (same trick as BYOL / V-JEPA).

Pipeline:
  frames (B, L, 1, 64, 64)
    -> context_encoder (per-frame, grads flow)         -> (B, L, 128, 8, 8)
    -> causal temporal attention (the predictor)        -> (B, L, 128, 8, 8)
    -> compare predicted[:, :-1] against target_encoder(frames)[:, 1:]  (stop-grad)

target_encoder is a frame-level encoder only — no temporal mixing on that side,
by design (it just needs to produce the thing being predicted, not predict it).
"""

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

try:
    from .encoder import JEPAEncoderPair
    from ..attention import MultiHeadAttention, PositionalEncoding
    from ..encoder_decoder import FFN
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from jepa.encoder import JEPAEncoderPair
    from attention import MultiHeadAttention, PositionalEncoding
    from encoder_decoder import FFN


class JEPATR(nn.Module):
    """
    JEPA encoder pair + causal temporal-attention predictor.

    The predictor block (attention_1/ffn_1/attention_2/ffn_2) is reused
    verbatim from HPCPredictiveCoder's _apply_temporal_attention — same
    2-block, post-norm, causally-masked structure — so this is a controlled
    ablation against the VPC: same encoder capacity, same predictor capacity,
    only the loss location (latent space, not pixels) and target source
    (EMA encoder, not ground-truth pixels) differ.

    Args:
        in_channels: Input channels (1 for depth)
        layers:      Residual blocks per encoder stage [2, 2, 2, 2]
        latent_dim:  Latent channel dim (128 -> 8x8 patches at 64x64 input)
        num_heads:   Attention heads for the predictor
        dropout:     Positional encoding dropout (position is instantiated
                     but never applied — matches HPCPredictiveCoder exactly,
                     see model.py's _apply_temporal_attention)

    Input:  (B, L, C, 64, 64)
    """

    def __init__(
        self,
        in_channels: int = 1,
        layers: List[int] = [2, 2, 2, 2],
        latent_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.encoders = JEPAEncoderPair(in_channels, layers, latent_dim)

        self.attention_1 = MultiHeadAttention(latent_dim, num_heads)
        self.ffn_1 = FFN(latent_dim, latent_dim)

        self.attention_2 = MultiHeadAttention(latent_dim, num_heads)
        self.ffn_2 = FFN(latent_dim, latent_dim)

        # Instantiated, never called — matches HPCPredictiveCoder's dead
        # positional encoding exactly, so predictor capacity is identical
        # between the VPC and JEPATR (see model.py:75, never used in
        # _apply_temporal_attention).
        self.position = PositionalEncoding(latent_dim, dropout)

    def train(self, mode: bool = True) -> "JEPATR":
        super().train(mode)
        self.encoders.target_encoder.eval()
        return self

    def _predict(self, f: Tensor) -> Tensor:
        """
        Causal self-attention over context latents — the predictor.
        f: (B, L, C, H, W) -> (B, L, C, H, W), position i only attends to <= i.
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

        return f

    def _encode_context_sequence(self, x: Tensor) -> Tensor:
        """x: (B, L, C, H, W) -> (B, L, latent_dim, h, w), gradients flow."""
        B, L, C, H, W = x.shape
        z = self.encoders.encode_context(x.reshape(B * L, C, H, W))
        return z.reshape(B, L, z.shape[1], z.shape[2], z.shape[3])

    @torch.no_grad()
    def _encode_target_sequence(self, x: Tensor) -> Tensor:
        """x: (B, L, C, H, W) -> (B, L, latent_dim, h, w), stop-gradient, no temporal mixing."""
        B, L, C, H, W = x.shape
        z = self.encoders.encode_target(x.reshape(B * L, C, H, W))
        return z.reshape(B, L, z.shape[1], z.shape[2], z.shape[3])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, L, C, 64, 64)
        Returns:
            predicted: (B, L-1, latent_dim, 8, 8) — predictor output at position i
                       (saw frames 1..i causally), trained to match frame i+1's target
            target:    (B, L-1, latent_dim, 8, 8) — stop-gradient target latents,
                       frames 2..L (i.e. shifted by one to line up with predicted)
        """
        z_ctx = self._encode_context_sequence(x)
        predicted = self._predict(z_ctx)

        z_tgt = self._encode_target_sequence(x)

        return predicted[:, :-1], z_tgt[:, 1:]

    def get_latents(self, x: Tensor) -> Tensor:
        """
        Predictor output for every position, using full causal context —
        the JEPA analog of HPCPredictiveCoder.get_latents(): the temporally-
        integrated representation to feed into extract_latents.py-style
        spatial/Pearson/MI analysis, not the frame-level encoder output alone.

        Args:
            x: (B, L, C, 64, 64)
        Returns:
            (B, L, latent_dim, 8, 8)
        """
        z_ctx = self._encode_context_sequence(x)
        return self._predict(z_ctx)

    def update_target_encoder(self, momentum: float) -> None:
        self.encoders.update_target_encoder(momentum)


if __name__ == "__main__":
    print("Testing JEPATR...")

    model = JEPATR(in_channels=1, latent_dim=128, num_heads=4)
    x = torch.randn(2, 10, 1, 64, 64)

    predicted, target = model(x)
    print(f"Input:      {tuple(x.shape)}")
    print(f"Predicted:  {tuple(predicted.shape)}  requires_grad={predicted.requires_grad}")
    print(f"Target:     {tuple(target.shape)}  requires_grad={target.requires_grad}")

    latents = model.get_latents(x)
    print(f"get_latents: {tuple(latents.shape)}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total:,}")
    print(f"Trainable params: {trainable:,} (excludes frozen target_encoder)")

    # --- gradient flow sanity check ---
    print("\nTesting gradient flow...")
    loss = F.smooth_l1_loss(predicted, target)
    loss.backward()

    ctx_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoders.context_encoder.parameters())
    pred_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.attention_1.parameters())
    tgt_has_grad = any(p.grad is not None for p in model.encoders.target_encoder.parameters())

    print(f"loss: {loss.item():.6f}")
    print(f"context_encoder received gradients: {ctx_has_grad} (expected True)")
    print(f"predictor (attention_1) received gradients: {pred_has_grad} (expected True)")
    print(f"target_encoder received gradients: {tgt_has_grad} (expected False)")
