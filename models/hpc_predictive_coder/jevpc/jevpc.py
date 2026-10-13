"""
JEVPC — Joint-Embedding Visual Predictive Coder.

Hybrid of VPC (explicit pixel-reconstruction predictive coding) and JEPATR
(implicit EMA/stop-gradient latent predictive coding), combining what each
was found to be good/bad at this session:

  - VPC (reconstruction-forced) encodes topology well -- distance-in-latent-
    space tracks distance-in-physical-space closely, because reconstructing
    every pixel forces the encoder to retain a comprehensive scene map. But
    it has no explicit pressure toward *viewpoint invariance*: approaching
    the same physical place from a very different angle (e.g. 180 degrees
    rotated) produces a very different image, and nothing in a pure
    reconstruction objective tells the encoder "this is still the same
    place" -- it just reconstructs whatever it sees.
  - JEPATR (EMA/slow-feature) does the opposite: because there is no pixel
    target, the encoder is free to collapse genuinely-different-looking but
    nearby/overlapping views toward similar latents (the "slowness"
    principle) -- the robustness VPC lacks. But every JEPATR variant this
    session (even with the predictor-level ordinal rank loss, see
    project_jepatr.md Run 8) has trailed VPC substantially on the actual
    topology metric (Pearson r vs physical distance) -- the thing reconstruction
    gets "for free," EMA alone has never matched.

JEVPC's bet: put both objectives on the SAME predictor output, so one
representation has to satisfy both constraints simultaneously.

Architecture: JEPAEncoderPair (context + EMA target, unchanged, reused
directly from jepa/encoder.py) -> causal temporal-attention predictor (same
2x MHA+FFN block design as VPC/JEPATR/CPC's predictors) -> two heads on the
predictor's output:
  1. HPCDecoder (reused from encoder_decoder.py, same as VPC/autoencoder) ->
     L1/MSE pixel reconstruction loss against the next frame (VPC's loss).
  2. Direct comparison against the EMA target encoder's latent (JEPATR's
     jepa_loss: normalized smooth-L1 + variance + covariance regularization).

Per the architecture discussion: the predictor-level ordinal temporal-rank
loss (JEPATR Run 8's big win) is kept, applied to the predictor's full
output -- the attention block doesn't get any direct supervision from
either the reconstruction or JEPA losses about temporal *ordering*, only
about per-position correctness, so the rank loss still adds something
neither other term provides. The encoder-level ("conv aux style") rank loss
is dropped -- reconstruction now directly anchors the encoder's quality, so
that auxiliary pressure on the raw per-frame encoding is no longer needed.
"""

import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

try:
    from ..encoder_decoder import HPCDecoder, FFN
    from ..attention import MultiHeadAttention, PositionalEncoding
    from ..jepa.encoder import JEPAEncoderPair
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from encoder_decoder import HPCDecoder, FFN
    from attention import MultiHeadAttention, PositionalEncoding
    sys.path.insert(0, str(Path(__file__).parent.parent / 'jepa'))
    from encoder import JEPAEncoderPair


class JEVPC(nn.Module):
    """
    Args:
        in_channels:  Input/output channels (1=depth, 3=RGB)
        layers:       Residual blocks per encoder/decoder stage [2,2,2,2]
        latent_dim:   Latent channel dim (128 -> 8x8 patches at 64x64 input,
                      matching VPC/JEPATR/CPC's matched-capacity convention)
        num_heads:    Attention heads for the predictor

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

        # stem_channels / num_stages: width-scaled encoder+decoder ("option C"); num_blocks: predictor depth.
        # Defaults reproduce the original model exactly (same state_dict keys).
        self.encoders = JEPAEncoderPair(in_channels, layers, latent_dim, stem_channels, num_stages)
        self.decoder = HPCDecoder(latent_dim, in_channels, layers[::-1], stem_channels, num_stages)

        self.attention_1 = MultiHeadAttention(latent_dim, num_heads)
        self.ffn_1 = FFN(latent_dim, latent_dim)

        self.attention_2 = MultiHeadAttention(latent_dim, num_heads)
        self.ffn_2 = FFN(latent_dim, latent_dim)

        assert num_blocks >= 2
        self.extra_attention = nn.ModuleList([MultiHeadAttention(latent_dim, num_heads) for _ in range(num_blocks - 2)])
        self.extra_ffn = nn.ModuleList([FFN(latent_dim, latent_dim) for _ in range(num_blocks - 2)])

        # Instantiated, never called -- matches JEPATR/HPCPredictiveCoder/CPC's
        # dead positional encoding, so predictor capacity stays identical
        # across every model in the comparison.
        self.position = PositionalEncoding(latent_dim, dropout)

    def train(self, mode: bool = True) -> "JEVPC":
        super().train(mode)
        self.encoders.target_encoder.eval()
        return self

    def _predict(self, f: Tensor) -> Tensor:
        """Causal self-attention over context latents -- the predictor.
        Identical block to VPC/JEPATR/CPC's _predict/_apply_temporal_attention."""
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
            residual = f
            f = F.layer_norm(ffn(f) + residual, f.shape[1:])

        return f

    def _encode_context_sequence(self, x: Tensor) -> Tensor:
        """x: (B, L, C, H, W) -> (B, L, latent_dim, h, w), gradients flow."""
        B, L, C, H, W = x.shape
        z = self.encoders.encode_context(x.reshape(B * L, C, H, W))
        return z.reshape(B, L, z.shape[1], z.shape[2], z.shape[3])

    @torch.no_grad()
    def _encode_target_sequence(self, x: Tensor) -> Tensor:
        """x: (B, L, C, H, W) -> (B, L, latent_dim, h, w), stop-gradient."""
        B, L, C, H, W = x.shape
        z = self.encoders.encode_target(x.reshape(B * L, C, H, W))
        return z.reshape(B, L, z.shape[1], z.shape[2], z.shape[3])

    def _decode(self, f: Tensor) -> Tensor:
        """f: (B, L, latent_dim, h, w) -> (B, L, C, H, W)."""
        B, L, C, H, W = f.shape
        out = self.decoder(f.reshape(B * L, C, H, W))
        return out.reshape(B, L, out.shape[1], out.shape[2], out.shape[3])

    def forward(self, x: Tensor, return_full: bool = False):
        """
        Args:
            x: (B, L, C, 64, 64)
            return_full: also return (a) the raw per-frame context-encoder
                output z_ctx (B, L, latent_dim, 8, 8), and (b) the predictor's
                full unshifted output predicted_full (B, L, latent_dim, 8, 8)
                -- the same tensor get_latents() returns -- for the
                predictor-level temporal-rank loss.
        Returns:
            reconstructed: (B, L-1, C, 64, 64) -- decoder(predicted[:, :-1]),
                       trained via L1/MSE against x[:, 1:] (VPC's loss,
                       next-frame reconstruction, matching vpc/train.py's
                       predictions[:, :-1] vs sequences[:, 1:] convention)
            predicted:  (B, L-1, latent_dim, 8, 8) -- predictor output at
                       position i, trained to match frame i+1's EMA target
            target:    (B, L-1, latent_dim, 8, 8) -- stop-gradient EMA target
                       latents, frames 2..L
            z_ctx, predicted_full (only if return_full=True)
        """
        z_ctx = self._encode_context_sequence(x)
        predicted_full = self._predict(z_ctx)

        z_tgt = self._encode_target_sequence(x)

        predicted = predicted_full[:, :-1]
        target = z_tgt[:, 1:]
        reconstructed = self._decode(predicted)

        if return_full:
            return reconstructed, predicted, target, z_ctx, predicted_full
        return reconstructed, predicted, target

    def get_latents(self, x: Tensor) -> Tensor:
        """
        Predictor output (before the decoder) for every position, using full
        causal context -- the JEVPC analog of VPC/JEPATR/CPC's get_latents(),
        feeding extract_latents.py-style downstream analysis.

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
    print("Testing JEVPC...")

    model = JEVPC(in_channels=1, latent_dim=128, num_heads=4)
    x = torch.randn(2, 10, 1, 64, 64)

    reconstructed, predicted, target = model(x)
    print(f"Input:        {tuple(x.shape)}")
    print(f"Reconstructed:{tuple(reconstructed.shape)}  requires_grad={reconstructed.requires_grad}")
    print(f"Predicted:    {tuple(predicted.shape)}  requires_grad={predicted.requires_grad}")
    print(f"Target:       {tuple(target.shape)}  requires_grad={target.requires_grad}")

    reconstructed, predicted, target, z_ctx, predicted_full = model(x, return_full=True)
    print(f"z_ctx:        {tuple(z_ctx.shape)}")
    print(f"predicted_full:{tuple(predicted_full.shape)}")

    latents = model.get_latents(x)
    print(f"get_latents:  {tuple(latents.shape)}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total:,}  (trainable: {trainable:,})")

    # --- gradient flow sanity checks ---
    print("\nTesting gradient flow...")
    recon_loss = F.l1_loss(reconstructed, x[:, 1:])
    recon_loss.backward(retain_graph=True)
    enc_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoders.context_encoder.parameters())
    dec_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())
    pred_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.attention_1.parameters())
    print(f"recon_loss: {recon_loss.item():.6f}")
    print(f"encoder received grad from recon_loss: {enc_has_grad} (expected True)")
    print(f"decoder received grad from recon_loss: {dec_has_grad} (expected True)")
    print(f"predictor received grad from recon_loss: {pred_has_grad} (expected True)")

    target_enc_has_grad = any(p.grad is not None for p in model.encoders.target_encoder.parameters())
    print(f"target_encoder has .grad at all: {target_enc_has_grad} (expected False -- stop-gradient)")
