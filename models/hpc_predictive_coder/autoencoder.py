"""
HPCAutoencoder — encoder/decoder without temporal attention.

Same convolutional backbone as HPCPredictiveCoder but operates on
individual frames (no sequence dimension).

latent_dim=128: input (B, C, 64, 64)  → latent (B, 128, 8, 8) → output (B, C, 64, 64)
latent_dim=512: input (B, C, 256, 256) → latent (B, 512, 8, 8) → output (B, C, 256, 256)
"""

import torch
import torch.nn as nn

try:
    from .encoder_decoder import HPCEncoder, HPCDecoder
except ImportError:
    from encoder_decoder import HPCEncoder, HPCDecoder


class HPCAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = HPCEncoder(in_channels, layers=[2, 2, 2, 2], latent_dim=latent_dim)
        self.decoder = HPCDecoder(latent_dim, in_channels, layers=[2, 2, 2, 2])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → latent: (B, latent_dim, h, w)"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim, h, w) → (B, C, H, W)"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


if __name__ == '__main__':
    model = HPCAutoencoder(in_channels=1, latent_dim=128)
    x = torch.randn(4, 1, 64, 64)
    out = model(x)
    z = model.encode(x)
    print(f'Input:   {x.shape}')
    print(f'Latent:  {z.shape}')
    print(f'Output:  {out.shape}')
    print(f'Params:  {sum(p.numel() for p in model.parameters()):,}')
