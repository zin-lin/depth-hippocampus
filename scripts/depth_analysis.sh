#!/bin/bash
# Extract depth latents for rec_02-06 + rec_wh_inf_01-03, then combined spatial analysis.
# Usage: ./scripts/depth_analysis.sh [--reso 64] [--dim 512] [--checkpoint path]
#                                     [--neighborhood 70] [--skip]
#
# --reso 64  : use depth64/ dirs + hpc_depth64_latest.pth (for 128x128 source recordings)
# --reso 256 : use depth/    dirs + hpc_predictive_coder_latest.pth (default)
#
# NOTE: rec_03-06 are 128x128 source depth — only work with --reso 64
#
# Per-recording offsets (stationary warmup frames):
#   rec_02=42  rec_03=45  rec_04=41  rec_05=42  rec_06=44
#   rec_wh_inf_01=44  rec_wh_inf_02=41  rec_wh_inf_03=44

set -e

RESO=256
DIM=512
CHECKPOINT=""
NEIGHBORHOOD=70
SKIP=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --reso)         RESO="$2";         shift 2 ;;
        --dim)          DIM="$2";          shift 2 ;;
        --checkpoint)   CHECKPOINT="$2";   shift 2 ;;
        --neighborhood) NEIGHBORHOOD="$2"; shift 2 ;;
        --skip)         SKIP=1;            shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Auto-select checkpoint based on resolution
if [ -z "$CHECKPOINT" ]; then
    if [ "$RESO" -eq 64 ]; then
        CHECKPOINT="models/hpc_predictive_coder/hpc_depth64_latest.pth"
        DIM=128
    else
        CHECKPOINT="models/hpc_predictive_coder/hpc_predictive_coder_latest.pth"
        DIM=512
    fi
fi

# Depth directory name
if [ "$RESO" -eq 64 ]; then
    DEPTH_SUBDIR="depth64"
    LATENT_SUBDIR="latents_depth64"
else
    DEPTH_SUBDIR="depth"
    LATENT_SUBDIR="latents"
fi

# Recordings with per-recording offsets (associative array)
declare -A OFFSETS
OFFSETS["rec_02"]=42
OFFSETS["rec_03"]=45
OFFSETS["rec_04"]=41
OFFSETS["rec_05"]=42
OFFSETS["rec_06"]=44
OFFSETS["rec_wh_inf_01"]=44
OFFSETS["rec_wh_inf_02"]=41
OFFSETS["rec_wh_inf_03"]=44

RECS=("rec_02" "rec_03" "rec_04" "rec_05" "rec_06" "rec_wh_inf_01" "rec_wh_inf_02" "rec_wh_inf_03")

echo "=================================="
echo "Mode:         depth (${RESO}x${RESO})"
echo "Depth dir:    ${DEPTH_SUBDIR}/"
echo "Latent dim:   ${DIM}"
echo "Checkpoint:   ${CHECKPOINT}"
echo "Neighborhood: ${NEIGHBORHOOD}"
echo "Datasets:     ${RECS[*]}"
echo "=================================="

LATENT_PATHS=()
POSITION_PATHS=()

for REC in "${RECS[@]}"; do
    REC_DIR="data/${REC}"
    DEPTH_DIR="${REC_DIR}/${DEPTH_SUBDIR}"
    ODOM="${REC_DIR}/odometry.npz"
    OUTPUT_DIR="${REC_DIR}/${LATENT_SUBDIR}"
    OFFSET="${OFFSETS[$REC]}"

    if [ ! -d "$DEPTH_DIR" ]; then
        echo "SKIP ${REC}: ${DEPTH_SUBDIR}/ not found"
        continue
    fi
    if [ ! -f "$ODOM" ]; then
        echo "SKIP ${REC}: odometry.npz not found"
        continue
    fi

    if [ "$SKIP" -eq 1 ]; then
        if [ ! -f "${OUTPUT_DIR}/latents_flat.npy" ]; then
            echo "SKIP ${REC}: latents not found at ${OUTPUT_DIR}, run without --skip first"
            continue
        fi
        echo "  [skip] ${REC} (offset=${OFFSET})"
    else
        echo -e "\n[Extract] ${REC} (offset=${OFFSET})..."
        python3 models/hpc_predictive_coder/extract_latents.py \
            --checkpoint "${CHECKPOINT}" \
            --depth_dir  "${DEPTH_DIR}" \
            --odom_file  "${ODOM}" \
            --output_dir "${OUTPUT_DIR}" \
            --latent_dim "${DIM}"
    fi

    LATENT_PATHS+=("${OUTPUT_DIR}/latents_flat.npy")
    POSITION_PATHS+=("${OUTPUT_DIR}/positions.npy")
done

if [ ${#LATENT_PATHS[@]} -eq 0 ]; then
    echo "No valid recordings found. Exiting."
    exit 1
fi

OUTPUT="data/depth${RESO}_combined_spatial_analysis.png"

echo -e "\n[Spatial analysis] ${#LATENT_PATHS[@]} recordings combined..."
python3 models/hpc_predictive_coder/spatial_analysis.py \
    --latents   "${LATENT_PATHS[@]}" \
    --positions "${POSITION_PATHS[@]}" \
    --neighborhood "${NEIGHBORHOOD}" \
    --output "${OUTPUT}" \
    --raw

echo -e "\nDone. Results saved to ${OUTPUT%.*}_linear.png and ${OUTPUT%.*}_log.png"
