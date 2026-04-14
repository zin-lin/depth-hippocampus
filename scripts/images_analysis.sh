#!/bin/bash
# Extract RGB latents for rec_02-06, then combined spatial analysis.
# Usage: ./scripts/images_analysis.sh [--reso 64] [--dim 128] [--checkpoint path]
#                                      [--neighborhood 70] [--skip]
#
# NOTE: rec_wh_inf_01-03 are excluded — no odometry_i.npz recorded for those.
#
# Per-recording offsets (stationary warmup frames):
#   rec_02=42  rec_03=45  rec_04=41  rec_05=42  rec_06=44

set -e

RESO=64
DIM=128
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

if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT="models/hpc_predictive_coder/hpc_rgb_latest.pth"
fi

if [ "$RESO" -eq 64 ]; then
    IMG_SUBDIR="images64"
else
    IMG_SUBDIR="images"
fi
LATENT_SUBDIR="latents_rgb"

declare -A OFFSETS
OFFSETS["rec_02"]=42
OFFSETS["rec_03"]=45
OFFSETS["rec_04"]=41
OFFSETS["rec_05"]=42
OFFSETS["rec_06"]=44

RECS=("rec_02" "rec_03" "rec_04" "rec_05" "rec_06")

echo "=================================="
echo "Mode:         images (${RESO}x${RESO})"
echo "Image dir:    ${IMG_SUBDIR}/"
echo "Latent dim:   ${DIM}"
echo "Checkpoint:   ${CHECKPOINT}"
echo "Neighborhood: ${NEIGHBORHOOD}"
echo "Datasets:     ${RECS[*]}"
echo "=================================="

LATENT_PATHS=()
POSITION_PATHS=()

for REC in "${RECS[@]}"; do
    REC_DIR="data/${REC}"
    IMG_DIR="${REC_DIR}/${IMG_SUBDIR}"
    ODOM="${REC_DIR}/odometry_i.npz"
    OUTPUT_DIR="${REC_DIR}/${LATENT_SUBDIR}"
    OFFSET="${OFFSETS[$REC]}"

    if [ ! -d "$IMG_DIR" ]; then
        echo "SKIP ${REC}: ${IMG_SUBDIR}/ not found"
        continue
    fi
    if [ ! -f "$ODOM" ]; then
        echo "SKIP ${REC}: odometry_i.npz not found"
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
            --rgb_dir    "${IMG_DIR}" \
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

OUTPUT="data/rgb${RESO}_combined_spatial_analysis.png"

echo -e "\n[Spatial analysis] ${#LATENT_PATHS[@]} recordings combined..."
python3 models/hpc_predictive_coder/spatial_analysis.py \
    --latents   "${LATENT_PATHS[@]}" \
    --positions "${POSITION_PATHS[@]}" \
    --neighborhood "${NEIGHBORHOOD}" \
    --output "${OUTPUT}" \
    --raw

echo -e "\nDone. Results saved to ${OUTPUT%.*}_linear.png and ${OUTPUT%.*}_log.png"
