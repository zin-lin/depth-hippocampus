#!/usr/bin/env python3
"""Untrained (random-init) checkpoints, seeds 0-4, for the map-test floor (VPC, TwoHead, V-JEPA architectures,
3-channel). Each is built exactly as the evaluation builds it and saved with zero training steps.
Writes results/random_ckpts/{vpc_rgb,twohead_rgb,vjepa}_seed<s>.pth"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RANDOM_CKPTS, _cls
create_hpc_predictive_coder = _cls('vpc/model.py', 'create_hpc_predictive_coder')
JEVPCTwoHead = _cls('jevpc_twohead/jevpc_twohead.py', 'JEVPCTwoHead')
VJEPA = _cls('vjepa/vjepa.py', 'VJEPA')

RANDOM_CKPTS.mkdir(parents=True, exist_ok=True)
for seed in range(5):
    torch.manual_seed(seed)
    m = create_hpc_predictive_coder(seq_len=10, in_channels=3, out_channels=3, num_heads=4, latent_dim=128)
    torch.save({'epoch': -1, 'model_state_dict': m.state_dict()}, RANDOM_CKPTS / f'vpc_rgb_seed{seed}.pth')
    torch.manual_seed(seed)
    m = JEVPCTwoHead(in_channels=3, latent_dim=128, num_heads=4)
    torch.save({'epoch': -1, 'model_state_dict': m.state_dict(), 'in_channels': 3, 'latent_dim': 128, 'num_heads': 4,
                'seq_len': 10}, RANDOM_CKPTS / f'twohead_rgb_seed{seed}.pth')
    torch.manual_seed(seed)
    m = VJEPA(encoder='hybrid')
    torch.save({'epoch': -1, 'model_state_dict': m.state_dict(), 'in_channels': 3, 'dim': 128, 'depth': None, 'heads': 4,
                'pred_dim': 64, 'pred_depth': 4, 'pred_heads': 4, 'encoder': 'hybrid', 'seq_len': 10},
               RANDOM_CKPTS / f'vjepa_seed{seed}.pth')
    torch.manual_seed(seed)
    m = VJEPA(encoder='vit')
    torch.save({'epoch': -1, 'model_state_dict': m.state_dict(), 'in_channels': 3, 'dim': 128, 'depth': None, 'heads': 4,
                'pred_dim': 64, 'pred_depth': 4, 'pred_heads': 4, 'encoder': 'vit', 'seq_len': 10},
               RANDOM_CKPTS / f'vjepa_vit_seed{seed}.pth')
print(sorted(p.name for p in RANDOM_CKPTS.iterdir()))
