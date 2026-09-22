"""
JEPA encoder pair for depth predictive coding.

Context encoder is trained by gradient descent; target encoder is an
EMA (exponential moving average) of the context encoder's weights,
updated after every optimizer step: target = m * target + (1 - m) * context.
The target encoder never receives gradients directly.

latent_dim=128, input 64x64 -> latent (128, 8, 8), 64 spatial patches.
Matches the paper's small-image feature-extraction focus (comparable to
hpc_depth64.pth) rather than the coarser 2x2 grid the 512-dim/64x64
Depth-VPC checkpoint produces.
"""

import copy
import sys
from pathlib import Path
from typing import List

import torch
from torch import nn, Tensor

try:
    from ..encoder_decoder import HPCEncoder
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from encoder_decoder import HPCEncoder


class JEPAEncoderPair(nn.Module):
    """
    Holds a context encoder (trained via backprop) and a target encoder
    (stop-gradient, weights copied from the context encoder).

    Args:
        in_channels: Input channels (1 for depth)
        layers:      Residual blocks per encoder stage [2, 2, 2, 2]
        latent_dim:  Latent channel dim (128 -> 8x8 patches at 64x64 input,
                     matching hpc_depth64.pth's resolution/grid)

    Input:  (B, C, 64, 64)
    Output: (B, latent_dim, 8, 8)
    """

    def __init__(
        self,
        in_channels: int = 1,
        layers: List[int] = [2, 2, 2, 2],
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder = HPCEncoder(in_channels, layers, latent_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        # HPCEncoder uses BatchNorm2d throughout (residual.py). Keep the target
        # encoder permanently in eval mode so its BN layers normalize with their
        # running stats instead of updating them from the target branch's own
        # forward passes — the running stats are synced from the context encoder
        # explicitly in update_target_encoder() instead.
        self.target_encoder.eval()

    def train(self, mode: bool = True) -> "JEPAEncoderPair":
        super().train(mode)
        self.target_encoder.eval()
        return self

    def encode_context(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W) -> (B, latent_dim, h, w), gradients flow."""
        return self.context_encoder(x)

    @torch.no_grad()
    def encode_target(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W) -> (B, latent_dim, h, w), stop-gradient."""
        return self.target_encoder(x)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """
        EMA update: target_encoder = momentum * target_encoder + (1 - momentum) * context_encoder.


        BatchNorm running stats (running_mean, running_var, num_batches_tracked)
        are copied directly rather than EMA'd again, since they're already a
        running average computed by BN itself from the context encoder's data.
        """
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1 - momentum)

        for b_ctx, b_tgt in zip(self.context_encoder.buffers(), self.target_encoder.buffers()):
            b_tgt.data.copy_(b_ctx.data)


if __name__ == "__main__":
    print("Testing JEPAEncoderPair...")

    pair = JEPAEncoderPair(in_channels=1, latent_dim=128)
    pair.train()
    x = torch.randn(2, 1, 64, 64)

    z_ctx = pair.encode_context(x)
    z_tgt = pair.encode_target(x)

    print(f"Input:          {x.shape}")
    print(f"Context latent: {tuple(z_ctx.shape)}  requires_grad={z_ctx.requires_grad}")
    print(f"Target latent:  {tuple(z_tgt.shape)}  requires_grad={z_tgt.requires_grad}")
    print(f"target_encoder.training = {pair.target_encoder.training} (should be False even though pair.train() was called)")

    ctx_params = sum(p.numel() for p in pair.context_encoder.parameters())
    tgt_params = sum(p.numel() for p in pair.target_encoder.parameters())
    print(f"Context params: {ctx_params:,}")
    print(f"Target params:  {tgt_params:,} (frozen, EMA-updated)")

    # --- EMA update sanity check ---
    print("\nTesting EMA update...")
    p_ctx0 = next(pair.context_encoder.parameters()).clone()
    p_tgt0 = next(pair.target_encoder.parameters()).clone()
    assert torch.equal(p_ctx0, p_tgt0), "context/target should start identical"

    # Simulate an optimizer step perturbing the context encoder only.
    with torch.no_grad():
        for p in pair.context_encoder.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    momentum = 0.99
    pair.update_target_encoder(momentum)

    p_ctx1 = next(pair.context_encoder.parameters())
    p_tgt1 = next(pair.target_encoder.parameters())
    expected_tgt1 = momentum * p_tgt0 + (1 - momentum) * p_ctx1

    print(f"context moved:            {(p_ctx1 - p_ctx0).abs().mean().item():.6f} (mean abs delta)")
    print(f"target moved:             {(p_tgt1 - p_tgt0).abs().mean().item():.6f} (mean abs delta, should be ~1% of context's)")
    print(f"target matches EMA formula: {torch.allclose(p_tgt1, expected_tgt1, atol=1e-6)}")
    print(f"target still != context:  {not torch.equal(p_tgt1, p_ctx1)}")
