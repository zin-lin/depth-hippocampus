"""
Residual blocks for depth image predictive coding.

Implements ResNet-style residual blocks with optional downsampling/upsampling
for the encoder and decoder networks.
"""

from typing import List, Union

import torch
import torch.nn.functional as F
from torch import nn


class Residual(nn.Module):
    """
    Basic residual block with optional downsampling.

    Args:
        in_channels: Number of input channels
        channels: Number of output channels
        downsample: Whether to apply 2x downsampling (stride=2)
    """
    def __init__(
        self,
        in_channels: int,
        channels: int,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.downsample = downsample

        if downsample:
            self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1)
            self.skip = nn.Conv2d(in_channels, channels, kernel_size=1, stride=2)
        else:
            self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out


class ResidualTranspose(nn.Module):
    """
    Transposed residual block with optional upsampling.

    Args:
        in_channels: Number of input channels
        channels: Number of output channels
        upsample: Whether to apply 2x upsampling
    """
    def __init__(
        self,
        in_channels: int,
        channels: int,
        upsample: bool = False
    ) -> None:
        super().__init__()
        self.upsample = upsample

        self.skip = None
        if in_channels < channels:
            self.skip = nn.ConvTranspose2d(in_channels, channels, kernel_size=1, stride=1)
        elif in_channels > channels:
            delta_channels = in_channels - channels
            self.skip = lambda x: torch.cat(
                [x, torch.zeros(x.shape[0], delta_channels, x.shape[2], x.shape[3], device=x.device)],
                dim=1
            )

        if upsample:
            self.conv1 = nn.ConvTranspose2d(
                in_channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1
            )
            self.skip = nn.ConvTranspose2d(in_channels, channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.ConvTranspose2d(in_channels, channels, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out


class DownBlock2d(nn.Module):
    """
    Stack of residual blocks with optional initial downsampling.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        layers: Number of residual blocks in this stage
        downsample: Whether to apply downsampling in the first block
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layers: int,
        downsample: bool = False
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Residual(in_channels, out_channels, downsample=downsample)
        ])
        for _ in range(layers - 1):
            self.layers.append(Residual(out_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class UpBlock2d(nn.Module):
    """
    Stack of transposed residual blocks with optional initial upsampling.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        layers: Number of residual blocks in this stage
        upsample: Whether to apply upsampling in the first block
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layers: int,
        upsample: bool = False
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualTranspose(in_channels, out_channels, upsample=upsample)
        ])
        for _ in range(layers - 1):
            self.layers.append(ResidualTranspose(out_channels, out_channels))

    def forward(self, x: torch.Tensor, skip: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """
        Forward pass with optional skip connection.

        Args:
            x: Input tensor
            skip: Optional skip connection from encoder (not used in current architecture)
        """
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x
