"""
Temporal attention mechanisms for depth image sequences.

Implements multi-head attention with causal masking for autoregressive
prediction of depth frames.
"""

from typing import Tuple, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for temporal sequences of spatial feature maps.

    Projects the queries, keys and values through learned convolutions,
    then performs attention over the temporal dimension while preserving
    spatial structure.

    Args:
        d_model: Feature dimension (must be divisible by num_heads)
        num_heads: Number of attention heads (default: 2)

    Input shape: (batch, seq_len, channels, height, width)
    Output shape: (batch, seq_len, channels, height, width)
    """

    def __init__(self, d_model: int = 512, num_heads: int = 2):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.d_model = d_model

        self.query_proj = nn.Conv2d(
            d_model, self.d_head * num_heads, kernel_size=1, bias=False
        )
        self.key_proj = nn.Conv2d(
            d_model, self.d_head * num_heads, kernel_size=1, bias=False
        )
        self.value_proj = nn.Conv2d(
            d_model, self.d_head * num_heads, kernel_size=1, bias=False
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Compute multi-head attention.

        Args:
            query: Query tensor (B, L, C, H, W)
            key: Key tensor (B, L, C, H, W)
            value: Value tensor (B, L, C, H, W)
            mask: Optional causal mask (L, L)

        Returns:
            context: Attended features (B, L, C, H, W)
            attn: Always None — torch's fused SDPA kernel never materializes attention weights
        """
        batch_size = value.size(0)
        seq_len = value.size(1)
        channels = value.size(2)
        height = value.size(3)
        width = value.size(4)

        # Project Q, K, V through convolutions
        query = query.reshape(-1, channels, height, width)
        query = self.query_proj(query)
        query = query.reshape(
            batch_size, seq_len, self.num_heads, self.d_head, height, width
        )

        key = key.reshape(-1, channels, height, width)
        key = self.key_proj(key)
        key = key.reshape(
            batch_size, seq_len, self.num_heads, self.d_head, height, width
        )

        value = value.reshape(-1, channels, height, width)
        value = self.value_proj(value)
        value = value.reshape(
            batch_size, seq_len, self.num_heads, self.d_head, height, width
        )

        # Flatten spatial dimensions for temporal attention
        query = query.contiguous().view(
            batch_size, seq_len, self.num_heads, self.d_head * height * width
        )
        key = key.contiguous().view(
            batch_size, seq_len, self.num_heads, self.d_head * height * width
        )
        value = value.contiguous().view(
            batch_size, seq_len, self.num_heads, self.d_head * height * width
        )

        context = self._sdpa(query, key, value, mask)

        # Reshape back to spatial format
        context = context.view(
            batch_size, seq_len, self.num_heads * self.d_head, height, width
        )

        return context, None

    def _sdpa(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        torch's native fused scaled-dot-product attention (flash-attention backend
        on supported GPUs, falls back to an equivalent math kernel otherwise).

        Args:
            query: (B, L, H, D)
            key: (B, L, H, D)
            value: (B, L, H, D)
            mask: Optional mask (L, L), True = block that (query, key) position —
                  inverted internally, since torch's boolean attn_mask convention
                  is True = allowed to attend.

        Returns:
            context: Attended values (B, L, H, D)
        """
        # (B, L, H, D) -> (B, H, L, D), torch SDPA's expected layout
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_mask = None
        if mask is not None:
            attn_mask = ~mask.to(device=query.device, dtype=torch.bool)

        context = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)

        # (B, H, L, D) -> (B, L, H, D)
        return context.transpose(1, 2).contiguous()


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.

    Adds position information to feature maps using sine/cosine functions.
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (B, L, C, H, W)

        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
