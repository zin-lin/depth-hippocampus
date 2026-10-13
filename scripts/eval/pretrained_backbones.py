"""Frozen, pretrained robotics/vision backbones used as baselines (inference only).

Protocol (shared by all, matches the AE convention since none are temporal):
  - descriptor of the LAST frame of each window; depth is broadcast to 3 channels by the caller
  - 64x64 -> 224x224 bilinear upsample; each model's own input normalisation
  - descriptor L2-normalised (standard for place recognition)
Caveat for the paper: all were trained on >=224 px natural images; our frames are 64 px, upsampled.

Backbones:
  yolo         YOLO11m backbone (layers 0-10, COCO), last feature map global-average-pooled      -> 512-d
  frcnn        Faster R-CNN ResNet-50-FPN v2 backbone body (COCO), layer4 global-average-pooled   -> 2048-d
  dinov2       DINOv2 ViT-B/14 (HF facebook/dinov2-base), CLS token                              -> 768-d
  netvlad      VGG16 + NetVLAD, Pittsburgh30k weights (hloc distribution), PCA-whitened          -> 4096-d
  anyloc       DINOv2 ViT-B/14 layer-9 patch tokens + hard-assignment VLAD (K=32) with the vocabulary
               fitted on our TRAINING recordings rec_wh_01-04 of the same input modality         -> 24576-d
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
PRE = ROOT / 'results/pretrained'
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _prep(x, size=224, imagenet=True):
    """x: (B, 3, 64, 64) in [0, 1] -> upsampled (and ImageNet-normalised) tensor."""
    x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    if imagenet:
        x = (x - IMNET_MEAN.to(x.device)) / IMNET_STD.to(x.device)
    return x


# ------------------------------------------------------------------ YOLO11
def build_yolo(dev):
    from ultralytics import YOLO
    PRE.mkdir(parents=True, exist_ok=True)
    y = YOLO(str(PRE / 'yolo11m.pt'))
    layers = y.model.model[:11].to(dev).eval()          # backbone: Conv/C3k2/SPPF/C2PSA, all sequential (f=-1)
    def f(x):
        h = _prep(x, imagenet=False)                     # ultralytics models take [0, 1] RGB
        for m in layers:
            h = m(h)
        return F.normalize(h.mean(dim=(2, 3)), dim=1)
    return f


# ------------------------------------------------------------------ Faster R-CNN ResNet-50
def build_frcnn(dev):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    body = m.backbone.body.to(dev).eval()                # IntermediateLayerGetter -> {'0'..'3'}
    def f(x):
        return F.normalize(body(_prep(x))['3'].mean(dim=(2, 3)), dim=1)
    return f


# ------------------------------------------------------------------ DINOv2
def _dinov2(dev):
    from transformers import AutoModel
    return AutoModel.from_pretrained('facebook/dinov2-base').to(dev).eval()


def build_dinov2(dev):
    m = _dinov2(dev)
    def f(x):
        return F.normalize(m(pixel_values=_prep(x)).last_hidden_state[:, 0], dim=1)
    return f


# ------------------------------------------------------------------ NetVLAD (hloc port)
NETVLAD_URL = 'https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat'


class NetVLADLayer(nn.Module):
    def __init__(self, input_dim=512, K=64):
        super().__init__()
        self.score_proj = nn.Conv1d(input_dim, K, kernel_size=1, bias=False)
        self.centers = nn.Parameter(torch.empty(input_dim, K))
        self.output_dim = input_dim * K

    def forward(self, x):                                # x: (B, D, N)
        scores = F.softmax(self.score_proj(x), dim=1)    # (B, K, N)
        diff = x.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(-1)   # (B, D, K, N)
        desc = (scores.unsqueeze(1) * diff).sum(dim=-1)  # (B, D, K)
        desc = F.normalize(desc, dim=1)                  # intra-normalisation
        return F.normalize(desc.reshape(x.shape[0], -1), dim=1)


def build_netvlad(dev):
    from scipy.io import loadmat
    from torchvision import models
    PRE.mkdir(parents=True, exist_ok=True)
    path = PRE / 'Pitts30K_struct.mat'
    if not path.exists():
        torch.hub.download_url_to_file(NETVLAD_URL, str(path))
    mat = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    vgg = list(models.vgg16().children())[0]
    backbone = nn.Sequential(*list(vgg.children())[:-2])  # up to conv5_3 (drop final ReLU + max-pool)
    for layer, ml in zip(backbone.children(), mat['net'].layers):
        if isinstance(layer, nn.Conv2d):
            layer.weight = nn.Parameter(torch.tensor(ml.weights[0]).float().permute(3, 2, 0, 1))
            layer.bias = nn.Parameter(torch.tensor(ml.weights[1]).float())
    vlad = NetVLADLayer()
    vlad.score_proj.weight = nn.Parameter(torch.tensor(mat['net'].layers[30].weights[0]).float().permute(1, 0).unsqueeze(-1))
    vlad.centers = nn.Parameter(-torch.tensor(mat['net'].layers[30].weights[1]).float())
    whiten = nn.Linear(vlad.output_dim, 4096)
    whiten.weight = nn.Parameter(torch.tensor(mat['net'].layers[33].weights[0]).float().squeeze().permute(1, 0))
    whiten.bias = nn.Parameter(torch.tensor(mat['net'].layers[33].weights[1].squeeze()).float())
    mean = torch.tensor(np.array(mat['net'].meta.normalization.averageImage[0, 0], dtype=np.float32)).view(1, 3, 1, 1)
    scale = 255.0 if float(mean.max()) > 1.0 else 1.0    # MatConvNet mean is in [0,255] units
    backbone, vlad, whiten = backbone.to(dev).eval(), vlad.to(dev).eval(), whiten.to(dev).eval()
    mean = mean.to(dev)
    def f(x):
        h = _prep(x, imagenet=False) * scale - mean
        d = backbone(h)
        d = F.normalize(d.reshape(d.shape[0], d.shape[1], -1), dim=1)
        return F.normalize(whiten(vlad(d)), dim=1)
    return f


# ------------------------------------------------------------------ AnyLoc (DINOv2 ViT-B + VLAD)
ANYLOC_LAYER, ANYLOC_K = 9, 32
VOCAB_RECS = ['rec_wh_01', 'rec_wh_02', 'rec_wh_03', 'rec_wh_04']   # training recordings only


def _anyloc_tokens(m, x):
    out = m(pixel_values=_prep(x), output_hidden_states=True)
    return out.hidden_states[ANYLOC_LAYER][:, 1:]       # (B, 256 patches, 768), CLS dropped


def _vocab_frames(input_mode, n_per_rec=500):
    from PIL import Image
    fr = []
    for r in VOCAB_RECS:
        d = ROOT / 'data' / r
        if input_mode == 'depth':
            files = sorted((d / 'depth64').glob('*.npy'))
            pick = [files[i] for i in np.linspace(0, len(files) - 1, n_per_rec).astype(int)]
            a = np.stack([np.load(f).astype(np.float32) for f in pick])
            a = np.clip(np.where(np.isinf(a) | np.isnan(a), 20.0, a), 0, 20) / 20.0
            fr.append(np.repeat(a[:, None], 3, axis=1))
        else:
            files = sorted(list((d / 'images64').glob('*.jpg')) + list((d / 'images64').glob('*.png')))
            pick = [files[i] for i in np.linspace(0, len(files) - 1, n_per_rec).astype(int)]
            fr.append(np.stack([np.asarray(Image.open(f), dtype=np.float32).transpose(2, 0, 1) / 255.0 for f in pick]))
    return torch.from_numpy(np.concatenate(fr))


@torch.no_grad()
def _anyloc_vocab(m, input_mode, dev):
    path = PRE / f'anyloc_vitb_l{ANYLOC_LAYER}_k{ANYLOC_K}_{input_mode}.npy'
    if path.exists():
        return torch.from_numpy(np.load(path)).to(dev)
    from sklearn.cluster import KMeans
    fr = _vocab_frames(input_mode)
    toks = torch.cat([F.normalize(_anyloc_tokens(m, fr[b:b + 64].to(dev)), dim=-1).reshape(-1, 768).cpu()
                      for b in range(0, len(fr), 64)])
    sub = toks[torch.randperm(len(toks), generator=torch.Generator().manual_seed(0))[:200_000]].numpy()
    km = KMeans(n_clusters=ANYLOC_K, n_init=3, random_state=0).fit(sub)
    c = km.cluster_centers_.astype(np.float32)
    PRE.mkdir(parents=True, exist_ok=True); np.save(path, c)
    return torch.from_numpy(c).to(dev)


def build_anyloc(dev, input_mode):
    m = _dinov2(dev)
    centers = _anyloc_vocab(m, input_mode, dev)          # (K, 768)
    def f(x):
        t = F.normalize(_anyloc_tokens(m, x), dim=-1)    # (B, N, 768)
        assign = torch.cdist(t, centers.unsqueeze(0).expand(len(t), -1, -1)).argmin(-1)   # hard assignment
        onehot = F.one_hot(assign, ANYLOC_K).float()     # (B, N, K)
        resid = (onehot.unsqueeze(-1) * (t.unsqueeze(2) - centers.view(1, 1, ANYLOC_K, -1))).sum(1)   # (B, K, 768)
        resid = F.normalize(resid, dim=-1)               # intra-normalisation
        return F.normalize(resid.reshape(len(t), -1), dim=1)
    return f


BUILDERS = {'yolo': build_yolo, 'frcnn': build_frcnn, 'dinov2': build_dinov2, 'netvlad': build_netvlad}


def build(name, dev, input_mode):
    if name.endswith('_ft'):
        return build_finetuned(name[:-3], dev)
    if name == 'anyloc':
        return build_anyloc(dev, input_mode)
    return BUILDERS[name](dev)


# ------------------------------------------------------------------ label-free fine-tuned variants (scripts/finetune_vpr.py)
FT = ROOT / 'results/finetuned'


def build_finetuned(name, dev):
    """name: 'netvlad' or 'anyloc'. Loads results/finetuned/<name>_ft.pth (best val) into the same trainable
    architecture used for fine-tuning; AnyLoc-FT keeps its trained soft-assignment VLAD layer."""
    import sys as _s
    _s.path.insert(0, str(ROOT / 'scripts'))
    from finetune_vpr import build_trainable
    model, fwd = build_trainable(name, dev)
    model.load_state_dict(torch.load(FT / f'{name}_ft.pth', map_location=dev, weights_only=False)['model'])
    model.eval()
    return fwd
