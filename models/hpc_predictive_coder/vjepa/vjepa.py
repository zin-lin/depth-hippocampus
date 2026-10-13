"""
V-JEPA-style model, capacity-matched to VPC/JEVPC (~1.5M trainable params).

Differences from our temporal JEPATR/JEVPC, matching the V-JEPA recipe:
  - ViT encoders on spatio-temporal tokens instead of a CNN.
  - Context encoder sees only VISIBLE tokens; EMA target encoder (stop-grad) sees the full clip.
  - Predictor gets context tokens + learnable mask tokens at hidden positions and predicts the
    target encoder's (LayerNorm'd) embeddings there. Loss: L1 on masked tokens only.
  - Multi-block "tube" masking: the same spatial mask is applied to every time step.

Tokens: clip (B, 10, C, 64, 64) -> Conv3d k=s=(2,8,8) -> 5 x 8 x 8 = 320 tokens, dim 128.
Evaluation latent: target-encoder tokens of the last time step -> (128, 8, 8), the same 8192-dim
shape as VPC/JEVPC, so every existing eval script applies unchanged.

Simplification vs V-JEPA: one mask is sampled per BATCH (not per sample) so all samples share
the visible-token count.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Block(nn.Module):
    """Pre-LayerNorm transformer block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x: Tensor) -> Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class Encoder(nn.Module):
    def __init__(self, in_ch, dim, depth, heads, frames=10, img=64, patch=8, tubelet=2):
        super().__init__()
        self.grid = (frames // tubelet, img // patch, img // patch)
        self.n_tokens = self.grid[0] * self.grid[1] * self.grid[2]
        self.embed = nn.Conv3d(in_ch, dim, kernel_size=(tubelet, patch, patch), stride=(tubelet, patch, patch))
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def tokenize(self, x: Tensor) -> Tensor:
        """x: (B, T, C, H, W) -> (B, N, D) incl. position embedding."""
        t = self.embed(x.transpose(1, 2))            # (B, D, T', H', W')
        return t.flatten(2).transpose(1, 2) + self.pos

    def forward(self, x: Tensor, keep_idx: Tensor = None) -> Tensor:
        t = self.tokenize(x)
        if keep_idx is not None:
            t = t[:, keep_idx]
        for b in self.blocks:
            t = b(t)
        return self.norm(t)


class HybridEncoder(nn.Module):
    """Our ResNet encoder (HPCEncoder, same as VPC/JEVPC) per frame -> 8x8 grid, frame pairs averaged
    into tubelets -> 5x8x8 = 320 tokens + position embedding -> self-attention blocks.
    Masking happens in PIXEL space (masked patches zeroed before the ResNet) so convolution receptive
    fields can't leak hidden content into visible tokens."""

    def __init__(self, in_ch, dim, depth, heads, frames=10, img=64, patch=8, tubelet=2, layers=(2, 2, 2, 2)):
        super().__init__()
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from encoder_decoder import HPCEncoder
        self.cnn = HPCEncoder(in_ch, list(layers), dim)
        self.patch, self.tubelet = patch, tubelet
        self.grid = (frames // tubelet, img // patch, img // patch)
        self.n_tokens = self.grid[0] * self.grid[1] * self.grid[2]
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, keep_idx: Tensor = None) -> Tensor:
        B, T, C, H, W = x.shape
        Tp, Hp, Wp = self.grid
        if keep_idx is not None:                      # zero hidden patches in pixel space (tube mask)
            vis = torch.zeros(Tp * Hp * Wp, dtype=torch.bool, device=x.device)
            vis[keep_idx] = True
            spatial = vis.view(Tp, Hp, Wp)[0]         # tube mask: same at every time step
            pix = spatial.repeat_interleave(self.patch, 0).repeat_interleave(self.patch, 1)
            x = x * pix.to(x.dtype)
        f = self.cnn(x.reshape(B * T, C, H, W))       # (B*T, D, 8, 8)
        f = f.view(B, Tp, self.tubelet, f.shape[1], Hp, Wp).mean(2)          # tubelet pooling
        t = f.permute(0, 1, 3, 4, 2).reshape(B, self.n_tokens, -1) + self.pos
        if keep_idx is not None:
            t = t[:, keep_idx]
        for b in self.blocks:
            t = b(t)
        return self.norm(t)


class Predictor(nn.Module):
    def __init__(self, n_tokens, enc_dim, dim, depth, heads):
        super().__init__()
        self.inp = nn.Linear(enc_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, enc_dim)

    def forward(self, ctx: Tensor, keep_idx: Tensor, mask_idx: Tensor) -> Tensor:
        B = ctx.shape[0]
        c = self.inp(ctx) + self.pos[:, keep_idx]
        m = self.mask_token.expand(B, len(mask_idx), -1) + self.pos[:, mask_idx]
        x = torch.cat([c, m], dim=1)
        for b in self.blocks:
            x = b(x)
        return self.out(self.norm(x[:, c.shape[1]:]))        # predictions at masked positions


def sample_tube_mask(grid, generator=None, device='cpu', short_scale=0.25):
    """Multi-block tube mask on the (H', W') grid, repeated over all T' steps.
    Returns (keep_idx, mask_idx) as flat token indices into T'*H'*W'."""
    T, H, W = grid
    g = generator
    short = torch.rand(1, generator=g).item() < 0.5
    n_blocks, scale = (8, short_scale) if short else (2, 0.7)   # V-JEPA paper: short 8 x 0.15, long 2 x 0.7
    m = torch.zeros(H, W, dtype=torch.bool)
    for _ in range(n_blocks):
        area = scale * H * W
        ar = math.exp(torch.empty(1).uniform_(math.log(0.75), math.log(1.5), generator=g).item())
        h = max(1, min(H, int(round(math.sqrt(area * ar)))))
        w = max(1, min(W, int(round(math.sqrt(area / ar)))))
        top = torch.randint(0, H - h + 1, (1,), generator=g).item()
        left = torch.randint(0, W - w + 1, (1,), generator=g).item()
        m[top:top + h, left:left + w] = True
    if m.all():                                    # always keep at least a few visible patches
        m.view(-1)[torch.randperm(H * W, generator=g)[:4]] = False
    full = m.flatten().repeat(T)                   # tube: same spatial mask at every time step
    idx = torch.arange(T * H * W)
    return idx[~full].to(device), idx[full].to(device)


class VJEPA(nn.Module):
    def __init__(self, in_channels=3, dim=128, depth=None, heads=4, pred_dim=64, pred_depth=4, pred_heads=4,
                 frames=10, img=64, patch=8, tubelet=2, encoder='hybrid'):
        super().__init__()
        self.encoder_type = encoder
        if encoder == 'vit':
            self.context_encoder = Encoder(in_channels, dim, depth or 6, heads, frames, img, patch, tubelet)
        else:  # 'hybrid': our ResNet + self-attention; 3 blocks keeps total ~capacity-matched (~1.56M)
            self.context_encoder = HybridEncoder(in_channels, dim, depth or 3, heads, frames, img, patch, tubelet)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = Predictor(self.context_encoder.n_tokens, dim, pred_dim, pred_depth, pred_heads)
        self.grid = self.context_encoder.grid
        self.dim = dim

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()   # BatchNorm in the hybrid's ResNet: target uses running stats, as in JEPAEncoderPair
        return self

    def forward(self, x: Tensor, keep_idx: Tensor, mask_idx: Tensor):
        """Returns (pred, target) at masked positions, both (B, n_masked, dim)."""
        ctx = self.context_encoder(x, keep_idx)
        pred = self.predictor(ctx, keep_idx, mask_idx)
        with torch.no_grad():
            tgt = self.target_encoder(x)
            tgt = F.layer_norm(tgt, (tgt.shape[-1],))[:, mask_idx]
        return pred, tgt

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        for pt, pc in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.mul_(momentum).add_(pc.detach(), alpha=1 - momentum)

    @torch.no_grad()
    def get_latents(self, x: Tensor) -> Tensor:
        """Target-encoder tokens of the full clip, reshaped (B, T', dim, H', W'); [:, -1] = last step."""
        t = self.target_encoder(x)
        T, H, W = self.grid
        return t.view(x.shape[0], T, H, W, self.dim).permute(0, 1, 4, 2, 3)


if __name__ == "__main__":
    for enc in ('hybrid', 'vit'):
        print(f"===== {enc} =====")
        m = VJEPA(encoder=enc)
        m.train()
        x = torch.randn(2, 10, 3, 64, 64)
        keep, mask = sample_tube_mask(m.grid)
        pred, tgt = m(x, keep, mask)
        print(f"tokens={m.context_encoder.n_tokens} visible={len(keep)} masked={len(mask)}")
        print("pred", tuple(pred.shape), "target", tuple(tgt.shape), "latents", tuple(m.get_latents(x).shape))
        tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"trainable params: {tr:,} (VPC/JEVPC: ~1,511,681)  context enc: {sum(p.numel() for p in m.context_encoder.parameters()):,}  predictor: {sum(p.numel() for p in m.predictor.parameters()):,}")
        F.l1_loss(pred, tgt).backward()
        print("target encoder has grad:", any(p.grad is not None for p in m.target_encoder.parameters()), "(want False)")
        print("context encoder has grad:", any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.context_encoder.parameters()), "(want True)")
    if True:
        # leakage check (hybrid): changing ONLY hidden pixels must not change visible-token context features
        m = VJEPA(encoder='hybrid').eval()
        keep, mask = sample_tube_mask(m.grid)
        x = torch.randn(1, 10, 3, 64, 64); x2 = x.clone()
        T, H, W = m.grid
        vis = torch.zeros(T * H * W, dtype=torch.bool); vis[keep] = True
        hid = (~vis.view(T, H, W)[0]).repeat_interleave(8, 0).repeat_interleave(8, 1)
        x2[..., hid] += 5.0
        with torch.no_grad():
            d = (m.context_encoder(x, keep) - m.context_encoder(x2, keep)).abs().max().item()
        print(f"leakage check: max change in visible features when hidden pixels change = {d:.2e} (want 0)")
    ratios = [len(sample_tube_mask((5, 8, 8))[1]) / 320 for _ in range(1000)]
    print(f"mask ratio over 1000 samples: mean={sum(ratios) / 1000:.2f} min={min(ratios):.2f} max={max(ratios):.2f}")
