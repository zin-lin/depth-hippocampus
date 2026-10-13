#!/bin/bash
# Evaluate label-free fine-tuned NetVLAD/AnyLoc vs their frozen versions and our models (map, loop closure, perturbation).
cd "$(dirname "$0")/../.."
M=("VPC-RGB-{fed}-fp32" "TwoHead-RGB-{fed}-bf16" "NetVLAD-Pitts30k-{fed}" "AnyLoc-DINOv2B-{fed}" "NetVLAD-FT-RGB-{fed}" "AnyLoc-FT-RGB-{fed}")
for MODE in depth rgb; do
  F=$([ $MODE = depth ] && echo Depth || echo RGB)
  python3 scripts/eval/map_eval.py --input $MODE --stride 10 --models "${M[@]}" \
    --compare "NetVLAD-FT-RGB-$F" "NetVLAD-Pitts30k-$F" --compare "AnyLoc-FT-RGB-$F" "AnyLoc-DINOv2B-$F" \
    --compare "TwoHead-RGB-$F-bf16 MAP" "NetVLAD-FT-RGB-$F" --compare "TwoHead-RGB-$F-bf16 MAP" "AnyLoc-FT-RGB-$F"
  cp results/map_${MODE}_s10.txt results/map_${MODE}_s10_finetuned.txt
  python3 scripts/eval/loop_closure.py --recs wh --input $MODE --models "${M[@]}"
  cp results/loop_closure_wh_${MODE}_s10.txt results/loop_closure_wh_${MODE}_s10_finetuned.txt
  python3 scripts/eval/perturb_test.py --input $MODE --models "${M[@]}"
  cp results/perturb_${MODE}.txt results/perturb_${MODE}_finetuned.txt
done
echo FT_EVAL_DONE
