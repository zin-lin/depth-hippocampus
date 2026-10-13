"""
JEVPC-TwoHead -- MoE Stage 1: fork by function, not by sample.

Plain JEVPC puts reconstruction, JEPA and rank losses on ONE predictor
output, so that representation is a compromise between objectives. Here the
shared encoder output forks into two full predictors, each supervised ONLY
by its own loss:

    x -> shared encoder (context + EMA target) -> z_ctx
              |-> predictor_MAP -> h_MAP -> decoder -> recon loss
              '-> predictor_LC  -> h_LC  -> jepa_loss vs EMA target, rank loss

No gradient crosses between heads; only the encoder receives both. One
forward pass yields a VPC-like latent (h_MAP, for mapping) and a JEPA-like
latent (h_LC, for loop closure / localisation) -- usable with hand-coded
routing (map every window, localise every N windows) even if a learned MoE
gate on top never works.

predictor_MAP reuses JEVPC's own attention_1/ffn_1/attention_2/ffn_2;
predictor_LC is an identical, separately-parameterised copy.
"""
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

sys.path.insert(0, str(Path(__file__).parent.parent / 'jevpc'))
from jevpc import JEVPC

try:
    from ..encoder_decoder import FFN
    from ..attention import MultiHeadAttention
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from encoder_decoder import FFN
    from attention import MultiHeadAttention


class CausalPredictor(nn.Module):
    """Same 2x (MHA + FFN) causal block as JEVPC._predict, own parameters."""

    def __init__(self, latent_dim: int, num_heads: int, num_blocks: int = 2):
        super().__init__()
        self.extra_attention = nn.ModuleList([MultiHeadAttention(latent_dim, num_heads) for _ in range(num_blocks - 2)])
        self.extra_ffn = nn.ModuleList([FFN(latent_dim, latent_dim) for _ in range(num_blocks - 2)])
        self.attention_1 = MultiHeadAttention(latent_dim, num_heads)
        self.ffn_1 = FFN(latent_dim, latent_dim)
        self.attention_2 = MultiHeadAttention(latent_dim, num_heads)
        self.ffn_2 = FFN(latent_dim, latent_dim)

    def forward(self, f: Tensor) -> Tensor:
        L = f.shape[1]
        mask = torch.triu(torch.ones(L, L, device=f.device), diagonal=1).bool()

        residual = f
        f, _ = self.attention_1(f, f, f, mask)
        f = F.layer_norm(f + residual, f.shape[1:])
        f = F.layer_norm(self.ffn_1(f) + f, f.shape[1:])

        residual = f
        f, _ = self.attention_2(f, f, f, mask)
        f = F.layer_norm(f + residual, f.shape[1:])
        f = F.layer_norm(self.ffn_2(f) + f, f.shape[1:])

        for attn, ffn in zip(self.extra_attention, self.extra_ffn):   # optional extra blocks (num_blocks > 2)
            residual = f
            f, _ = attn(f, f, f, mask)
            f = F.layer_norm(f + residual, f.shape[1:])
            f = F.layer_norm(ffn(f) + f, f.shape[1:])
        return f


class JEVPCTwoHead(JEVPC):
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
        super().__init__(in_channels, layers, latent_dim, num_heads, dropout, num_blocks, stem_channels, num_stages)
        self.predictor_lc = CausalPredictor(latent_dim, num_heads, num_blocks)   # LC head: same depth as MAP head

    def forward(self, x: Tensor, x_clean: Tensor = None):
        """Returns (reconstructed, h_map_full, h_lc_full, target_full).
        reconstructed: decoder(h_map[:, :-1]) -- compare to x[:, 1:]
        h_lc_full / target_full: compare h_lc[:, :-1] to target[:, 1:]
        x_clean: optional uncorrupted clip for the EMA target encoder (augmentation runs: the context
        branch sees the corrupted x, the target sees clean frames). None -> target sees x, as before."""
        z_ctx = self._encode_context_sequence(x)
        h_map = self._predict(z_ctx)
        h_lc = self.predictor_lc(z_ctx)
        z_tgt = self._encode_target_sequence(x if x_clean is None else x_clean)
        reconstructed = self._decode(h_map[:, :-1])
        return reconstructed, h_map, h_lc, z_tgt

    def get_latents_both(self, x: Tensor):
        z_ctx = self._encode_context_sequence(x)
        return self._predict(z_ctx), self.predictor_lc(z_ctx)

    def get_latents(self, x: Tensor) -> Tensor:
        return self.get_latents_both(x)[0]


if __name__ == "__main__":
    m = JEVPCTwoHead(in_channels=1, latent_dim=128, num_heads=4)
    x = torch.randn(2, 10, 1, 64, 64)
    rec, hm, hl, tg = m(x)
    print("recon", tuple(rec.shape), "h_map", tuple(hm.shape), "h_lc", tuple(hl.shape), "target", tuple(tg.shape))
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,} (JEVPC: 1,511,681)")

    # gradient isolation check
    F.l1_loss(rec, x[:, 1:]).backward(retain_graph=True)
    lc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.predictor_lc.parameters())
    map_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.attention_1.parameters())
    print(f"recon -> predictor_MAP grad: {map_grad} (want True); -> predictor_LC grad: {lc_grad} (want False)")
    m.zero_grad()
    F.mse_loss(hl[:, :-1], tg[:, 1:]).backward()
    lc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.predictor_lc.parameters())
    map_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.attention_1.parameters())
    print(f"jepa  -> predictor_LC grad: {lc_grad} (want True); -> predictor_MAP grad: {map_grad} (want False)")
