"""
Encoder-Decoder architecture for HPC (High-Performance) predictive coding.

Deeper ResNet-style encoder/decoder than depth_predictive_coder.
512-dim latent at 8x8 spatial resolution for richer feature representation.
Follows double-channels-when-halve-spatial rule throughout.

Input: (B, C, 256, 256)
Encoder output: (B, 512, 8, 8)
"""

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .residual import DownBlock2d, UpBlock2d
except ImportError:
    from residual import DownBlock2d, UpBlock2d


class HPCEncoder(nn.Module):
    """
    ResNet-style encoder with configurable depth via latent_dim.

    latent_dim=128: 2-stage  → output (B, 128, 32, 32)
    latent_dim=512: 4-stage  → output (B, 512,  8,  8)

    Args:
        in_channels: Number of input channels (1 for depth, 3 for RGB)
        layers: Residual blocks per stage e.g. [2, 2, 2, 2]
        latent_dim: Output channel dim — 128 or 512
    """

    def __init__(self, in_channels: int = 1, layers: List[int] = [2, 2, 2, 2], latent_dim: int = 512,
                 stem_channels: int = 64, num_stages: int = None) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # num_stages None -> original behaviour (4 stages at latent 512, else 2). Width-scaled variant ("option C"):
        # num_stages=2 with a wide stem and any latent_dim keeps the /8 downsampling (8x8 grid at 64 px).
        if num_stages is None:
            num_stages = 4 if latent_dim == 512 else 2
        assert num_stages in (2, 4)
        if num_stages == 4:
            assert latent_dim == 512 and stem_channels == 64, '4-stage design is the original ACCM 512 encoder'
        w = stem_channels
        two_stage_out = latent_dim if num_stages == 2 else 128

        # Initial conv: H -> H/2
        self.conv1 = nn.Conv2d(in_channels, w, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(w)

        # MaxPool: H/2 -> H/4
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Shared first two stages
        self.down1 = DownBlock2d(w, w, layers[0])                              # no ds
        self.down2 = DownBlock2d(w, two_stage_out, layers[1], downsample=True)  # /2

        # Extra stages for the original 4-stage 512-dim design only
        if num_stages == 4:
            self.down3 = DownBlock2d(128, 256, layers[2], downsample=True)  # /2
            self.down4 = DownBlock2d(256, 512, layers[3], downsample=True)  # /2
        else:
            self.down3 = None
            self.down4 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.down1(x)
        x = self.down2(x)
        if self.down3 is not None:
            x = self.down3(x)
            x = self.down4(x)
        return x


class HPCDecoder(nn.Module):
    """
    Decoder mirroring HPCEncoder.

    latent_dim=128: 2-stage upsampling from (B, 128, 32, 32)
    latent_dim=512: 4-stage upsampling from (B, 512,  8,  8)

    Args:
        latent_dim: Should match encoder output channels (128 or 512)
        out_channels: Number of output channels (1 for depth, 3 for RGB)
        layers: Residual blocks per stage (reversed from encoder)
    """

    def __init__(
        self,
        latent_dim: int = 512,
        out_channels: int = 1,
        layers: List[int] = [2, 2, 2, 2],
        stem_channels: int = 64,
        num_stages: int = None
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        if num_stages is None:
            num_stages = 4 if latent_dim == 512 else 2
        w = stem_channels
        first_in = latent_dim if num_stages == 2 else 128   # 2-stage: decode straight from the latent

        if num_stages == 4:
            self.up1 = UpBlock2d(512, 256, layers[0], upsample=True)  # /8  -> /4
            self.up2 = UpBlock2d(256, 128, layers[1], upsample=True)  # /4  -> /2
        else:
            self.up1 = None
            self.up2 = None

        # Shared final stages
        self.up3 = UpBlock2d(first_in, w, layers[2], upsample=True)  # /2 -> /1 (relative to down2 out)
        self.up4 = UpBlock2d(w, w, layers[3], upsample=True)
        self.up5 = UpBlock2d(w, w, 1, upsample=True)

        self.conv_out = nn.ConvTranspose2d(w, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.up1 is not None:
            x = self.up1(x, None)
            x = self.up2(x, None)
        x = self.up3(x, None)
        x = self.up4(x, None)
        x = self.up5(x, None)
        x = self.conv_out(x)
        return x


class FFN(nn.Module):
    """
    Position-wise Feed-Forward Network for temporal sequences.
    Two 1x1 convs with ReLU nonlinearity in between.

    Args:
        in_channels: Input/output channel dimension
        out_channels: Output channel dimension
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, C, H, W)
        Returns:
            x: (B, L, C, H, W)
        """
        B, L, C, H, W = x.shape
        x = x.reshape(B * L, C, H, W)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = x.reshape(B, L, C, H, W)
        return x
