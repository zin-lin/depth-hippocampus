"""
HPC (High-Performance) Predictive Coder.

Deeper version of DepthPredictiveCoder with 512-dim latent at 8x8 spatial resolution.
Same temporal attention architecture, richer feature representation.

Architecture:
1. HPCEncoder: 256x256 -> (512, 8, 8) — 4 downsampling stages
2. Temporal MHA x2 + FFN x2 with causal masking
3. HPCDecoder: (512, 8, 8) -> 256x256

Compared to DepthPredictiveCoder:
- 128-dim -> 512-dim latent
- 32x32 -> 8x8 spatial grid
- 2 extra CNN stages (ResNet-50 style depth)
- Flattened latent for distance: 512*8*8 = 32768 (same as depth PC! just redistributed)
"""

from typing import List
import torch
import torch.nn.functional as F
from torch import nn, Tensor

try:
    from .encoder_decoder import HPCEncoder, HPCDecoder, FFN
    from .attention import MultiHeadAttention, PositionalEncoding
except ImportError:
    from encoder_decoder import HPCEncoder, HPCDecoder, FFN
    from attention import MultiHeadAttention, PositionalEncoding


class HPCPredictiveCoder(nn.Module):
    """
    High-performance temporal predictive coding model.

    Args:
        in_channels:  Input channels (1=depth, 3=RGB)
        out_channels: Output channels (1=depth, 3=RGB)
        layers:       Residual blocks per encoder stage [2,2,2,2]
        seq_len:      Sequence length for causal mask
        latent_dim:   Latent channel dim (512)
        num_heads:    Attention heads (default 4 for 512-dim)
        dropout:      Positional encoding dropout

    Input:  (B, L, C, 256, 256)
    Output: (B, L, C, 256, 256)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        layers: List[int] = [2, 2, 2, 2],
        seq_len: int = 10,
        latent_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Encoder / Decoder
        self.encoder = HPCEncoder(in_channels, layers, latent_dim)
        self.decoder = HPCDecoder(latent_dim, out_channels, layers[::-1])

        # Temporal attention blocks (2 blocks, same as depth PC)
        self.attention_1 = MultiHeadAttention(latent_dim, num_heads)
        self.ffn_1 = FFN(latent_dim, latent_dim)

        self.attention_2 = MultiHeadAttention(latent_dim, num_heads)
        self.ffn_2 = FFN(latent_dim, latent_dim)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        )

        self.position = PositionalEncoding(latent_dim, dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _apply_temporal_attention(self, f: Tensor) -> Tensor:
        """Shared attention pipeline used by forward() and get_latents()."""
        residual = f
        f, _ = self.attention_1(f, f, f, self.mask)
        f = f + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f = self.ffn_1(f) + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f, _ = self.attention_2(f, f, f, self.mask)
        f = f + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f = self.ffn_2(f) + residual
        f = F.layer_norm(f, f.shape[1:])

        return f

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, C, 256, 256)
        Returns:
            (B, L, C, 256, 256)
        """
        B, L, C, H, W = x.shape

        x = x.reshape(B * L, C, H, W)
        f = self.encoder(x)                                          # (B*L, 512, 8, 8)
        f = f.reshape(B, L, f.shape[1], f.shape[2], f.shape[3])     # (B, L, 512, 8, 8)

        f = self._apply_temporal_attention(f)                        # (B, L, 512, 8, 8)

        f = f.reshape(B * L, f.shape[2], f.shape[3], f.shape[4])
        x = self.decoder(f)                                          # (B*L, C, 256, 256)
        x = x.reshape(B, L, C, H, W)

        return x

    def get_latents(self, x: Tensor) -> Tensor:
        """
        Extract latents after temporal attention.

        Args:
            x: (B, L, C, 256, 256)
        Returns:
            (B, L, 512, 8, 8)
        """
        B, L, C, H, W = x.shape

        x = x.reshape(B * L, C, H, W)
        f = self.encoder(x)
        f = f.reshape(B, L, f.shape[1], f.shape[2], f.shape[3])

        return self._apply_temporal_attention(f)

    def encode_sequence(self, x: Tensor) -> Tensor:
        """Encoder only, no temporal attention."""
        B, L, C, H, W = x.shape
        x = x.reshape(B * L, C, H, W)
        f = self.encoder(x)
        return f.reshape(B, L, f.shape[1], f.shape[2], f.shape[3])


def create_hpc_predictive_coder(
    seq_len: int = 10,
    in_channels: int = 1,
    out_channels: int = 1,
    num_heads: int = 4,
    latent_dim: int = 512
) -> HPCPredictiveCoder:
    return HPCPredictiveCoder(
        in_channels=in_channels,
        out_channels=out_channels,
        layers=[2, 2, 2, 2],
        seq_len=seq_len,
        latent_dim=latent_dim,
        num_heads=num_heads,
        dropout=0.0
    )


if __name__ == "__main__":
    print("Testing HPCPredictiveCoder...")

    model = create_hpc_predictive_coder(seq_len=10)
    x = torch.randn(2, 10, 1, 256, 256)

    print(f"Input:   {x.shape}")

    out = model(x)
    print(f"Output:  {out.shape}")

    latents = model.get_latents(x)
    print(f"Latents: {latents.shape}")

    flat = latents[0, -1].reshape(-1)
    print(f"Flat latent for distance: {flat.shape}")  # should be 512*8*8 = 32768

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {total:,}")
