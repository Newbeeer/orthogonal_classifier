#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "run.sh <bias1>"
  exit 1
fi
wd_value=1e-4
bias1=$1

CUDA_VISIBLE_DEVICES=3 python -m train_disc.py   --data_dir=~/domainbed \
--algorithm ERM  --dataset ColoredMNIST  --trial_seed 0 --bias 0.6 0. --model_save \
--save_path stage1_${bias1} --epochs 1

CUDA_VISIBLE_DEVICES=3 python -m train_disc_weight_p.py   --data_dir=~/domainbed \
--algorithm ERM  --dataset ColoredMNIST  --trial_seed 0 --bias 0.6 ${bias1} --model_save \
--resume_path stage1_${bias1} --save_path stage2_${bias1}_reweight --epochs 3

CUDA_VISIBLE_DEVICES=3 python -m train_eval_weight_p.py  --data_dir=~/domainbed \
--algorithm ERM  --dataset ColoredMNIST --trial_seed 0 --bias 0.6 ${bias1}