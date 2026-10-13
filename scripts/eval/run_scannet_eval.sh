#!/bin/bash
# ScanNet (option A: map test on 6 large rooms; option C: real-sensor perturbation check). Data stays on /mnt/lexar.
cd "$(dirname "$0")/../.."
M=("VPC-RGB-{fed}-fp32" "TwoHead-RGB-{fed}-bf16" "NetVLAD-FT-RGB-{fed}" "AnyLoc-FT-RGB-{fed}")
for MODE in depth rgb; do
  F=$([ $MODE = depth ] && echo Depth || echo RGB)
  EXTRA=(); [ $MODE = depth ] && EXTRA=("VPC-Depth-{fed}-fp32")
  python3 scripts/eval/map_eval.py --recs scannet --input $MODE --stride 10 --floor --models "${M[@]}" "${EXTRA[@]}" \
    --compare "TwoHead-RGB-$F-bf16 MAP" "VPC-RGB-$F-fp32" --compare "TwoHead-RGB-$F-bf16 MAP" "NetVLAD-FT-RGB-$F" \
    --compare "TwoHead-RGB-$F-bf16 MAP" "AnyLoc-FT-RGB-$F"
  python3 scripts/eval/perturb_test.py --recs scannet --input $MODE --models "${M[@]}" "${EXTRA[@]}"
done
echo SCANNET_EVAL_DONE
