# Hippocampus: small self-supervised predictive models for cognitive maps

PhD project (Edinburgh Napier University). The project asks whether a **small** network, trained **label-free** on a robot's own camera stream, can learn a latent space whose geometry mirrors physical space. The idea follows the hippocampus: place-cell-like codes, and "nearby in latent = nearby in the world".

The starting point is ACCM (predictive coding learns cognitive maps, RGB, discrete Minecraft). This project moves that idea to a **continuous, metric-scale warehouse** (Gazebo, ~1,500 m²), to **depth and RGB**, and compares predictive coding with modern self-supervised objectives (JEPA, V-JEPA, CPC) and with large pretrained place-recognition networks.

> Status: active research, results are pre-publication. Target venue IROS 2027, then an extended journal article.

**Author:** Zin Htun, PhD candidate, Edinburgh Napier University

## Authorship and attribution

- **Architectures and research design are my own work and my responsibility.** That covers the model architectures (VPC, TwoHead / JEVPC fork-by-function, JEPATR, and the width-scaled variants), the experimental design, the evaluation protocol and the interpretation of the results.
- **The building blocks are standard components from the published literature:**
  - transformer and attention blocks;
  - reconstruction losses, as used in predictive coding and in ACCM;
  - JEPA / V-JEPA latent prediction with an EMA target;
  - CPC's InfoNCE;
  - VICReg-style variance and covariance regularisers;
  - NetVLAD and AnyLoc for the baselines.

  They are used as described in the original papers.
- **AI assistance.** An AI model (Claude, Anthropic) helped write code templates for these implementations, and for the evaluation and training scripts. All design decisions, and responsibility for the code and results, rest with the author.

## Models

All in-house models share one ResNet encoder (64×64 input → 128×8×8 latent) and a small causal temporal-attention predictor. Clips are 10 frames.

| Model | Training signal | Size (trainable) | Folder |
|---|---|---|---|
| **VPC** (visual predictive coder, ACCM-style) | next-frame pixel reconstruction | 1.5M | [models/hpc_predictive_coder/vpc/](models/hpc_predictive_coder/vpc/) |
| **TwoHead** (JEVPC, fork-by-function) | shared encoder → MAP head (reconstruction) + LC head (JEPA vs EMA target + temporal rank) | 1.7M (+0.7M frozen EMA) | [models/hpc_predictive_coder/jevpc_twohead/](models/hpc_predictive_coder/jevpc_twohead/) |
| JEVPC | reconstruction + JEPA + rank on one predictor | 1.5M | [models/hpc_predictive_coder/jevpc/](models/hpc_predictive_coder/jevpc/) |
| JEPATR | latent prediction vs EMA target + temporal rank | ~1.5M | [models/hpc_predictive_coder/jepa/](models/hpc_predictive_coder/jepa/) |
| CPC | InfoNCE next-latent + temporal rank | 0.86M | [models/hpc_predictive_coder/cpc/](models/hpc_predictive_coder/cpc/) |
| V-JEPA (hybrid and ViT) | masked-tubelet latent prediction, V-JEPA paper recipe | 1.5M | [models/hpc_predictive_coder/vjepa/](models/hpc_predictive_coder/vjepa/) |
| Autoencoder | single-frame reconstruction (no temporal context) | 1.35M | [models/hpc_predictive_coder/autoencoder/](models/hpc_predictive_coder/autoencoder/) |

Model names in the tables follow `Model-TrainData-FedData-Precision`. For example, `TwoHead-RGB-Depth-bf16` means trained on RGB, evaluated on depth, bf16 training.

**Baselines.** The baselines are NetVLAD (VGG16, Pitts30k, ~15M) and AnyLoc (DINOv2 ViT-B layer 9 + VLAD, ~86M). Both are fine-tuned on our warehouse RGB with the same label-free signal our models get: a temporal triplet loss, with positives within ±20 frames and negatives more than 2,000 frames apart. Only their last layers are trained ([scripts/finetune_vpr.py](scripts/finetune_vpr.py)). These are the "-FT" rows below.

## Evaluation

- **Map quality.** On 9 held-out recordings (Set B), we compute Pearson correlation between latent distance and **log** physical distance, as in ACCM, plus the mutual information between latent and position. Every learned model is compared with its own **random-init floor**: the same architecture with untrained weights, averaged over 5 seeds. That shows how much of the score comes from training rather than architecture. Pairs are compared with paired t and Wilcoxon tests over the 9 recordings.
- **Loop closure.** Retrieval of earlier visits (within 2 m) on rec_wh_05/06: R@1, AP, alias@1 (top-1 match is a wrong place) and median error. Descriptors use cross-fit PCA whitening to 1,024 dimensions, except NetVLAD, which has its own whitening.
- **Perturbation test.** Map-quality retention under sensor noise, blur, occlusion, pixel dropout and scale change.

Scripts: [scripts/eval/](scripts/eval/) ([map_eval.py](scripts/eval/map_eval.py), [loop_closure.py](scripts/eval/loop_closure.py), [perturb_test.py](scripts/eval/perturb_test.py)). Raw outputs: [results/](results/).

## Results

### 1. Map quality: our models beat both their random floors and the fine-tuned VPR networks on depth

All models were trained on RGB and are evaluated on **depth** input, so this is cross-sensor transfer. Values are means over 9 recordings; per-recording values are in [results/map_depth_s10_finetuned.txt](results/map_depth_s10_finetuned.txt).

| Model | Params | Pearson (log) | MI |
|---|---|---|---|
| **TwoHead-RGB-Depth-bf16 (MAP head)** | 1.7M | **0.760** | **0.700** |
| VPC-RGB-Depth-fp32 | 1.5M | 0.723 | 0.665 |
| NetVLAD-FT-RGB-Depth | ~15M | 0.742 | 0.551 |
| AnyLoc-FT-RGB-Depth | ~86M | 0.737 | 0.606 |
| V-JEPA-ViT-RGB-Depth-bf16 | 1.5M | 0.646 | 0.482 |
| *Random-init floor, TwoHead arch (5 seeds)* | | *0.618 ± 0.011* | *0.575 ± 0.011* |
| *Random-init floor, VPC arch (5 seeds)* | | *0.599 ± 0.015* | *0.571 ± 0.010* |

- **TwoHead MAP beats NetVLAD-FT on MI** by +0.149, winning 8 of 9 recordings (t p = 0.030, Wilcoxon p = 0.027). It does so with about 9× fewer parameters, and fine-tuning made NetVLAD worse on depth.
- **The models clearly beat their random floors:**
  - TwoHead: Pearson +0.142, 8 of 9 recordings (p = 0.001);
  - VPC: Pearson +0.124, 8 of 9 (p = 0.001).
- **V-JEPA (paper recipe) didn't collapse, but learned little map structure.** Its Pearson is only +0.04 above its own floor, and its MI is below the floor. Masked latent prediction alone gives a weak metric map at this scale.

### 2. Map quality on RGB input: large fine-tuned networks win

On RGB, fine-tuned NetVLAD and AnyLoc are clearly ahead: Pearson 0.843 for both, against 0.604 for TwoHead. They win all 9 recordings ([results/map_rgb_s10_finetuned.txt](results/map_rgb_s10_finetuned.txt)). Our advantage is specific to the cross-sensor (RGB → depth) setting and to model size, not universal.

### 3. Loop closure: retrieval is not where these latents shine

Results with depth input, 1,024-dimension whitening ([results/loop_closure_wh_depth_s10_white1024.txt](results/loop_closure_wh_depth_s10_white1024.txt)):

| Model | R@1 | AP | alias@1 | median error (m) |
|---|---|---|---|---|
| VPC-RGB-Depth-fp32 | 0.114 | 0.193 | 0.585 | 19.8 |
| TwoHead-RGB-Depth-bf16 MAP | 0.119 | 0.178 | 0.596 | 20.7 |
| NetVLAD-FT-RGB-Depth | 0.142 | 0.298 | 0.549 | 17.7 |
| AnyLoc-FT-RGB-Depth | 0.143 | 0.255 | 0.517 | 16.2 |

Chance R@1 is about 0.01. Everything is well above chance, but the VPR networks lead on AP.

The warehouse is highly repetitive, with identical racks and colours, so it's a good test of metric mapping and a hard, somewhat unfair test of retrieval. Retrieval-specific heads, and an environment with more distinctive places, are future work.

### 4. Robustness: predictive coding latents are the most stable

Depth input; "retention" is the perturbed score divided by the clean score; full table in [results/perturb_depth.txt](results/perturb_depth.txt).

| Model | Pearson retention: noise / blur / occlude / dropout | Latent shift under noise (lower is better) |
|---|---|---|
| VPC-RGB-Depth-fp32 | 0.99 / 0.98 / 0.92 / 0.94 | **0.12** |
| TwoHead-RGB-Depth-bf16 MAP | 0.94 / 0.99 / **0.95** / 0.74 | 0.50 |
| NetVLAD-FT-RGB-Depth | 0.95 / 0.99 / 0.75 / 0.87 | 1.01 |
| AnyLoc-FT-RGB-Depth | 0.92 / 1.04 / 0.83 / 0.92 | 1.04 |

- **VPC's latents barely move** under noise, about 8× less than the VPR networks.
- **TwoHead is the most robust to occlusion.**
- **Pixel dropout is TwoHead's known weakness.** Training with augmentation fixes it, but costs some clean map quality.

### 5. Known limitation: zero-shot transfer to real indoor scans

Run zero-shot, with no training on ScanNet, our warehouse-trained models fall to their random-init floor on six real ScanNet rooms. The fine-tuned big models transfer somewhat better ([results/map_depth_s10_scannet.txt](results/map_depth_s10_scannet.txt)).

The claim is therefore **small, label-free, in-situ learning**: the model learns the map of the environment it's deployed in. It isn't a universal pretrained backbone. Training on real scans is in progress.

## In progress

- **Scaling study:** width-scaled 512-dimensional variants of every model (VPC512, TwoHead512, CPC512, AE512, V-JEPA-ViT512, about 16–32M parameters each), to separate the effect of the objective from the effect of size.
- **First RGB-trained CPC.**
- **Multiple training seeds** for the headline models.
- **ScanNet and Habitat-Sim (HM3D):** training and held-out-room evaluation on real and photoreal indoor data.

## Repository layout

```
models/hpc_predictive_coder/   model code: shared encoder/decoder + attention, one folder per model
scripts/eval/                  evaluation suite (map, loop closure, perturbation, baselines)
scripts/finetune_vpr.py        label-free fine-tuning of NetVLAD / AnyLoc
scripts/train_with_autorestart.sh   crash-resilient local training wrapper
HPC_module/                    Slurm scripts for the university cluster
results/                       evaluation outputs (text tables)
```

Data (Gazebo recordings) and weights are not tracked in git. ScanNet is used under its Terms of Use, and none of its data is included in this repository.

## Setup

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
python3 models/hpc_predictive_coder/vpc/train.py --help
python3 scripts/eval/map_eval.py --help
```
