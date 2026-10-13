#!/bin/bash
# Label-free temporal fine-tuning of NetVLAD then AnyLoc on RGB rec_wh_01-06, each under the auto-restart watchdog.
cd "$(dirname "$0")/.."
for m in netvlad anyloc; do
  scripts/train_with_autorestart.sh /tmp/finetune_${m}.log results/finetuned/${m}_ft_latest.pth -- \
    python3 scripts/finetune_vpr.py --model $m --save results/finetuned/${m}_ft.pth --steps 3000
done
echo FINETUNE_ALL_DONE >> /tmp/finetune_anyloc.log
