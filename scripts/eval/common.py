"""Shared core for the evaluation suite (scripts/eval/).

Conventions (all agreed with the user, see memory):
  - Latent = a_L: the LAST window position's output, (128, 8, 8) -> flat 8192. AE (no temporal model) = encoder
    output of the window's last frame.
  - Windows: stride = seq_len (10/10 default, 20/20 for the long-window check).
  - Depth frames -> odometry.npz ; RGB frames -> odometry_i.npz.
  - Offsets: ceil((stationary warm-up frames + seq_len) / stride). Reproduces the stride-10 table and the verified
    stride-20 value (VPC-Depth rec_17 s20 = 0.7436). RGB warm-up differs by 1-2 frames -> same offsets.
  - Model names: Model-TrainData-FedData-Precision; FedData is filled in from the --input mode.
Latents are cached under results/latents/<model>/<input>_s<stride>/<rec>/ (never /tmp: it is wiped on reboot).
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / 'models/hpc_predictive_coder'
W = ROOT / 'weights'
DATA = ROOT / 'data'
CACHE = ROOT / 'results/latents'
RANDOM_CKPTS = ROOT / 'results/random_ckpts'
sys.path.insert(0, str(ROOT / 'scripts'))          # compare_pearson, mutual_information


def _cls(relpath, attr):
    """Load a model class straight from its file (folders share names with their modules, e.g.
    jevpc_twohead/jevpc_twohead.py, so plain imports pick up the package instead)."""
    import importlib.util
    f = MODELS_DIR / relpath
    for d in (str(f.parent), str(MODELS_DIR)):       # the model files import siblings / parent-level modules
        if d not in sys.path:
            sys.path.append(d)
    name = 'evalmod_' + relpath.replace('/', '_').replace('.py', '')
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, f)
        mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return getattr(sys.modules[name], attr)

SET_B = ['rec_15', 'rec_16', 'rec_17', 'rec_18', 'rec_19', 'rec_20', 'rec_21', 'rec_23', 'rec_24']
WH = ['rec_wh_05', 'rec_wh_06']
WARMUP = {'rec_15': 123, 'rec_16': 83, 'rec_17': 116, 'rec_18': 116, 'rec_19': 89,
          'rec_20': 115, 'rec_21': 109, 'rec_23': 122, 'rec_24': 122}
NEIGHBORHOOD = 60


def offset(rec, stride, seq_len):
    """Recordings without a stationary warm-up (e.g. ScanNet) use 0 -> offset = first full window."""
    return math.ceil((WARMUP.get(rec, 0) + seq_len) / stride)


# ScanNet rooms passing the pre-registered rule (floor extent >= 2.0 m in x and y, >= 70 windows), linked as data/sn_*
SCANNET = sorted(p.name for p in DATA.glob('sn_scene*'))
RECSETS = {'set_b': SET_B, 'scannet': SCANNET}


def short(rec):
    return rec.replace('rec_', '').replace('sn_scene', '')[:7]


# ------------------------------------------------------------------ model registry
# name stem -> dict(kind, ckpt, ch, heads). The full model name is f"{stem}" with {fed} replaced by the input mode.
REGISTRY = {
    'VPC-Depth-{fed}-fp32':         dict(kind='vpc', ckpt=W / 'vpc_depth_sgd/vpc_depth.pth', ch=1),
    'VPC-RGB-{fed}-fp32':           dict(kind='vpc', ckpt=W / 'vpc_rgb_sgd_b32/vpc_rgb.pth', ch=3),
    'VPC512-RGB-{fed}-bf16':        dict(kind='vpc', ckpt=W / 'vpc_rgb_c512_bf16/vpc_rgb_c512.pth', ch=3),   # option C, 23.9M
    'AE-Depth-{fed}-fp32':          dict(kind='ae', ckpt=W / 'ae_depth_sgd/ae_depth.pth', ch=1),
    'AE-RGB-{fed}-fp32':            dict(kind='ae', ckpt=W / 'ae_rgb_sgd/ae_rgb.pth', ch=3),
    'CPC-Depth-{fed}-fp32':         dict(kind='cpc', ckpt=W / 'cpc_rank/cpc.pth', ch=1),
    'JEPATR-Depth-{fed}-fp32':      dict(kind='jepatr', ckpt=W / 'jepatr_pred_rank/jepatr.pth', ch=1),
    'JEVPC-Depth-{fed}-fp32':       dict(kind='jevpc', ckpt=W / 'jevpc/jevpc.pth', ch=1),
    'JEVPC-RGB-{fed}-fp32':         dict(kind='jevpc', ckpt=W / 'jevpc_rgb/jevpc_rgb.pth', ch=3),
    'TwoHead-RGB-{fed}-fp32':       dict(kind='twohead', ckpt=W / 'jevpc_twohead_rgb/jevpc_twohead.pth.epoch_200', ch=3),
    'TwoHead-RGB-{fed}-bf16':       dict(kind='twohead', ckpt=W / 'jevpc_twohead_rgb_bf16/jevpc_twohead.pth.epoch_200', ch=3),
    'TwoHead-RGB-{fed}-bf16-aug':   dict(kind='twohead', ckpt=W / 'jevpc_twohead_rgb_bf16_aug/jevpc_twohead.pth.epoch_200', ch=3),
    'VJEPA-Hybrid-RGB-{fed}-bf16':  dict(kind='vjepa', ckpt=W / 'vjepa_hybrid_rgb_bf16/vjepa.pth.epoch_200', ch=3),
    'VJEPA-ViT-RGB-{fed}-bf16':     dict(kind='vjepa', ckpt=W / 'vjepa_vit_rgb_bf16/vjepa.pth.epoch_200', ch=3),
    # HPC runs (HPC_module/train_hpc_rgb_baselines512.sh); final-epoch checkpoints, not best-val
    'CPC-RGB-{fed}-bf16':           dict(kind='cpc', ckpt=W / 'cpc_rgb_bf16/cpc.pth.epoch_200', ch=3),
    'CPC512-RGB-{fed}-bf16':        dict(kind='cpc', ckpt=W / 'cpc_rgb_c512_bf16/cpc.pth.epoch_200', ch=3),
    'AE512-RGB-{fed}-bf16':         dict(kind='ae', ckpt=W / 'ae_rgb_c512_bf16/ae_rgb_ep200.pth', ch=3),
    'VJEPA-ViT512-RGB-{fed}-bf16':  dict(kind='vjepa', ckpt=W / 'vjepa_vit_rgb_c512_bf16/vjepa.pth.epoch_200', ch=3),
}
for _s in range(5):   # random-init floors (map test ONLY -- see memory feedback_random_floor_scope)
    REGISTRY[f'Random-VPC-arch-s{_s}'] = dict(kind='vpc', ckpt=RANDOM_CKPTS / f'vpc_rgb_seed{_s}.pth', ch=3)
    REGISTRY[f'Random-TwoHead-arch-s{_s}'] = dict(kind='twohead', ckpt=RANDOM_CKPTS / f'twohead_rgb_seed{_s}.pth', ch=3)
    REGISTRY[f'Random-VJEPA-arch-s{_s}'] = dict(kind='vjepa', ckpt=RANDOM_CKPTS / f'vjepa_seed{_s}.pth', ch=3)
    REGISTRY[f'Random-VJEPAViT-arch-s{_s}'] = dict(kind='vjepa', ckpt=RANDOM_CKPTS / f'vjepa_vit_seed{_s}.pth', ch=3)

# frozen pretrained robotics/vision backbones (baselines; inference only -- see pretrained_backbones.py)
for _n, _stem in (('yolo', 'YOLO11m-COCO-{fed}'), ('frcnn', 'FasterRCNN-R50-COCO-{fed}'), ('dinov2', 'DINOv2-B-{fed}'),
                  ('netvlad', 'NetVLAD-Pitts30k-{fed}'), ('anyloc', 'AnyLoc-DINOv2B-{fed}'),
                  ('netvlad_ft', 'NetVLAD-FT-RGB-{fed}'), ('anyloc_ft', 'AnyLoc-FT-RGB-{fed}')):
    REGISTRY[_stem] = dict(kind='pretrained', name=_n, ckpt=None, ch=3)
PRETRAINED = [s for s, v in REGISTRY.items() if v['kind'] == 'pretrained']


def heads(stem):
    """Output names a model produces. TwoHead -> ' MAP' and ' LC' suffixes; everything else one output."""
    return [' MAP', ' LC'] if REGISTRY[stem]['kind'] == 'twohead' else ['']


def full_name(stem, fed):
    fed = {'depth': 'Depth', 'rgb': 'RGB'}.get(fed, fed)
    return stem.replace('{fed}', fed)


def load_model(stem, seq_len, dev, input_mode='depth'):
    """Returns fn(x: (B, L, C, 64, 64)) -> {head_suffix: a_L (B, 128, 8, 8)}."""
    spec = REGISTRY[stem]; kind = spec['kind']
    if kind == 'pretrained':                                  # frozen baseline: descriptor of the window's last frame
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pretrained_backbones import build
        f = build(spec['name'], dev, input_mode)
        return lambda x: {'': f(x[:, -1])}
    ck = torch.load(spec['ckpt'], map_location='cpu', weights_only=False)
    sd = ck['model_state_dict']
    if kind == 'vpc':
        create_hpc_predictive_coder = _cls('vpc/model.py', 'create_hpc_predictive_coder')
        # architecture from the checkpoint when present (newer trainer saves it); defaults = original small VPC
        m = create_hpc_predictive_coder(seq_len=seq_len, in_channels=spec['ch'], out_channels=spec['ch'],
                                        num_heads=ck.get('num_heads', 4), latent_dim=ck.get('latent_dim', 128),
                                        num_blocks=ck.get('num_blocks', 2), stem_channels=ck.get('stem_channels', 64),
                                        num_stages=ck.get('num_stages', None))
        m.load_state_dict(sd, strict=False)
        f = lambda m, x: {'': m.get_latents(x)[:, -1]}
    elif kind == 'ae':
        HPCAutoencoder = _cls('autoencoder/autoencoder.py', 'HPCAutoencoder')
        m = HPCAutoencoder(spec['ch'], ck.get('latent_dim', 128), ck.get('stem_channels', 64), ck.get('num_stages')); m.load_state_dict(sd)
        f = lambda m, x: {'': m.encode(x[:, -1])}
    elif kind == 'cpc':
        CPC = _cls('cpc/cpc.py', 'CPC')
        m = CPC(in_channels=spec['ch'], latent_dim=ck.get('latent_dim', 128), num_heads=ck.get('num_heads', 4), num_blocks=ck.get('num_blocks', 2),
                stem_channels=ck.get('stem_channels', 64), num_stages=ck.get('num_stages')); m.load_state_dict(sd, strict=False)
        f = lambda m, x: {'': m.get_latents(x)[:, -1]}
    elif kind == 'jepatr':
        JEPATR = _cls('jepa/jepatr.py', 'JEPATR')
        m = JEPATR(in_channels=spec['ch'], latent_dim=128, num_heads=4); m.load_state_dict(sd, strict=False)
        f = lambda m, x: {'': m.get_latents(x)[:, -1]}
    elif kind == 'jevpc':
        JEVPC = _cls('jevpc/jevpc.py', 'JEVPC')
        m = JEVPC(in_channels=spec['ch'], latent_dim=128, num_heads=4); m.load_state_dict(sd, strict=False)
        f = lambda m, x: {'': m.get_latents(x)[:, -1]}
    elif kind == 'twohead':
        JEVPCTwoHead = _cls('jevpc_twohead/jevpc_twohead.py', 'JEVPCTwoHead')
        m = JEVPCTwoHead(in_channels=spec['ch'], latent_dim=ck.get('latent_dim', 128), num_heads=ck.get('num_heads', 4),
                         num_blocks=ck.get('num_blocks', 2), stem_channels=ck.get('stem_channels', 64),
                         num_stages=ck.get('num_stages', None)); m.load_state_dict(sd)
        def f(m, x):
            a, b = m.get_latents_both(x)
            return {' MAP': a[:, -1], ' LC': b[:, -1]}
    elif kind == 'vjepa':
        VJEPA = _cls('vjepa/vjepa.py', 'VJEPA')
        m = VJEPA(in_channels=ck['in_channels'], dim=ck['dim'], depth=ck['depth'], heads=ck['heads'], pred_dim=ck['pred_dim'],
                  pred_depth=ck['pred_depth'], pred_heads=ck['pred_heads'], frames=seq_len, encoder=ck['encoder'])
        m.load_state_dict(sd)
        f = lambda m, x: {'': m.get_latents(x)[:, -1]}
    else:
        raise ValueError(kind)
    m = m.to(dev).eval()
    return lambda x: f(m, x)


# ------------------------------------------------------------------ frames
def load_frames(rec, input_mode):
    """input_mode: 'depth' (1ch, or broadcast to 3ch by the caller) or 'rgb'. Returns (frames (N, C, 64, 64) float32
    in [0, 1], positions (N, 2))."""
    if input_mode == 'depth':
        files = sorted((DATA / rec / 'depth64').glob('*.npy'))
        d = np.stack([np.load(f).astype(np.float32) for f in files])
        d = np.clip(np.where(np.isinf(d) | np.isnan(d), 20.0, d), 0, 20) / 20.0
        frames = d[:, None]
        odom = np.load(DATA / rec / 'odometry.npz')
    else:
        files = sorted(list((DATA / rec / 'images64').glob('*.jpg')) + list((DATA / rec / 'images64').glob('*.png')))
        frames = np.stack([np.asarray(Image.open(f), dtype=np.float32).transpose(2, 0, 1) / 255.0 for f in files])
        odom = np.load(DATA / rec / 'odometry_i.npz')
    pos = np.stack([odom['x'], odom['y']], 1).astype(np.float32)
    assert len(frames) == len(pos), f"{rec}: {len(frames)} frames vs {len(pos)} odometry"
    return frames, pos


def fit_channels(x, ch):
    """x: (B, L, c, H, W). Broadcast 1-channel depth to 3 channels for RGB-architecture models."""
    if x.shape[2] == ch:
        return x
    assert x.shape[2] == 1 and ch == 3, (x.shape, ch)
    return x.expand(-1, -1, 3, -1, -1)


@torch.no_grad()
def extract(fn, ch, frames, stride, seq_len, dev, bs=64):
    """frames: torch (N, c, 64, 64). Returns ({head: (n_windows, 8192) float32}, window last-frame indices)."""
    starts = list(range(0, len(frames) - seq_len + 1, stride)); out = {}
    for b in range(0, len(starts), bs):
        x = torch.stack([frames[s:s + seq_len] for s in starts[b:b + bs]]).to(dev)
        for k, z in fn(fit_channels(x, ch)).items():
            out.setdefault(k, []).append(z.reshape(len(z), -1).float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}, [s + seq_len - 1 for s in starts]


def latent_dir(name, input_mode, stride, rec):
    return CACHE / name.replace(' ', '_') / f'{input_mode}_s{stride}' / rec


def get_latents(stems, recs, input_mode, stride, dev=None, refresh=False):
    """Extract (or load cached) latents for every stem x rec. Returns {full output name: {rec: (lat, pos)}}."""
    dev = dev or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = stride
    res = {}
    todo = [s for s in stems if refresh or not all(latent_dir(full_name(s, input_mode) + h, input_mode, stride, r).joinpath('latents_flat.npy').exists()
                                                   for h in heads(s) for r in recs)]
    frames_cache = {}
    for stem in todo:
        fn = load_model(stem, seq_len, dev, input_mode); ch = REGISTRY[stem]['ch']
        for r in recs:
            if r not in frames_cache:
                fr, pos = load_frames(r, input_mode); frames_cache[r] = (torch.from_numpy(fr), pos)
            fr, pos = frames_cache[r]
            lat, last = extract(fn, ch, fr, stride, seq_len, dev)
            for h, z in lat.items():
                d = latent_dir(full_name(stem, input_mode) + h, input_mode, stride, r); d.mkdir(parents=True, exist_ok=True)
                np.save(d / 'latents_flat.npy', z); np.save(d / 'positions.npy', pos[last])
        del fn; torch.cuda.empty_cache()
    for stem in stems:
        for h in heads(stem):
            name = full_name(stem, input_mode) + h
            res[name] = {r: (np.load(latent_dir(name, input_mode, stride, r) / 'latents_flat.npy').astype(np.float32),
                             np.load(latent_dir(name, input_mode, stride, r) / 'positions.npy'))
                         for r in recs}
    return res
