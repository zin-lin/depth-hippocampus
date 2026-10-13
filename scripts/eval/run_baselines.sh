#!/bin/bash
# Robotics backbone baselines (frozen, pretrained) vs our models: map, loop closure, perturbation; depth + RGB input.
cd "$(dirname "$0")/../.."
OURS=("VPC-RGB-{fed}-fp32" "TwoHead-RGB-{fed}-bf16")
BASE=("YOLO11m-COCO-{fed}" "FasterRCNN-R50-COCO-{fed}" "DINOv2-B-{fed}" "NetVLAD-Pitts30k-{fed}" "AnyLoc-DINOv2B-{fed}")
for MODE in depth rgb; do
  FED=$([ $MODE = depth ] && echo Depth || echo RGB)
  CMP=()
  for b in "${BASE[@]}"; do CMP+=(--compare "TwoHead-RGB-${FED}-bf16 MAP" "${b/\{fed\}/$FED}"); done
  python3 scripts/eval/map_eval.py --input $MODE --stride 10 --floor --models "${OURS[@]}" "${BASE[@]}" "${CMP[@]}"
  cp results/map_${MODE}_s10.txt results/map_${MODE}_s10_baselines.txt
  python3 scripts/eval/loop_closure.py --recs wh --input $MODE --models "${OURS[@]}" "${BASE[@]}"
  cp results/loop_closure_wh_${MODE}_s10.txt results/loop_closure_wh_${MODE}_s10_baselines.txt
  python3 scripts/eval/perturb_test.py --input $MODE --models "${OURS[@]}" "${BASE[@]}"
  cp results/perturb_${MODE}.txt results/perturb_${MODE}_baselines.txt
done
echo ALL_BASELINES_DONE
