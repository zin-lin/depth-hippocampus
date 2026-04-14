#!/bin/bash
# Usage: ./latent_analysis.sh <rec_suffix> [--dim 512] [--mode depth] [--checkpoint path] [--skip]
# Example: ./latent_analysis.sh wh_inf_03 --dim 128 --mode rgb
# Example: ./latent_analysis.sh wh_inf_03 --dim 512 --mode depth --skip

set -e

# Defaults
DIM=512
MODE=depth
CHECKPOINT=""
SKIP=0

# First positional arg is the rec suffix
REC_SUFFIX=$1
shift

# Parse remaining flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --dim)   DIM="$2";        shift 2 ;;
        --mode)  MODE="$2";       shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --skip)       SKIP=1;          shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$REC_SUFFIX" ]; then
    echo "Usage: $0 <rec_suffix> [--dim 512] [--mode depth] [--checkpoint path]"
    exit 1
fi

REC="data/rec_${REC_SUFFIX}"
if [ "$MODE" = "rgb" ]; then
    ODOM="${REC}/odometry_i.npz"
else
    ODOM="${REC}/odometry.npz"
fi

if [ ! -d "$REC" ]; then
    echo "Error: $REC not found"
    exit 1
fi

if [ ! -f "$ODOM" ]; then
    echo "Error: $ODOM not found"
    exit 1
fi

# Auto-select checkpoint if not provided
if [ -z "$CHECKPOINT" ]; then
    if [ "$MODE" = "rgb" ]; then
        CHECKPOINT="models/hpc_predictive_coder/hpc_rgb_latest.pth"
    else
        CHECKPOINT="models/hpc_predictive_coder/hpc_depth64_latest.pth"
    fi
fi

# Set paths based on mode
if [ "$MODE" = "rgb" ]; then
    DATA_DIR="${REC}/images64"
    OUTPUT_DIR="${REC}/latents_rgb"
    MODE_FLAG="--rgb_dir ${DATA_DIR}"
else
    DATA_DIR="${REC}/depth64"
    OUTPUT_DIR="${REC}/latents"
    MODE_FLAG="--depth_dir ${DATA_DIR}"
fi

echo "=================================="
echo "Rec:        rec_${REC_SUFFIX}"
echo "Mode:       ${MODE}"
echo "Latent dim: ${DIM}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output:     ${OUTPUT_DIR}"
echo "=================================="

# Step 1: Extract latents
if [ "$SKIP" -eq 1 ]; then
    if [ ! -f "${OUTPUT_DIR}/latents_flat.npy" ]; then
        echo "Error: latents not found at ${OUTPUT_DIR}, run without --skip first"
        exit 1
    fi
    echo -e "\n[1/2] Skipping extraction, using existing latents in ${OUTPUT_DIR}"
else
    echo -e "\n[1/2] Extracting latents..."
    python3 models/hpc_predictive_coder/extract_latents.py \
        --checkpoint "${CHECKPOINT}" \
        ${MODE_FLAG} \
        --odom_file "${ODOM}" \
        --output_dir "${OUTPUT_DIR}" \
        --latent_dim "${DIM}"
fi

# Step 2: Spatial analysis
echo -e "\n[2/2] Running spatial analysis..."
python3 models/hpc_predictive_coder/spatial_analysis.py \
    --latents "${OUTPUT_DIR}/latents_flat.npy" \
    --positions "${OUTPUT_DIR}/positions.npy" \
    --output "${OUTPUT_DIR}/spatial_analysis.png" \
    --neighborhood 30 \
    --offset 52 \
    

echo -e "\nDone. Results saved to ${OUTPUT_DIR}/"
