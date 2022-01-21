#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "run.sh <bias1>"
  exit 1
fi
wd_value=1e-4
bias1=$1

CUDA_VISIBLE_DEVICES=3 python -m train_orth/train_disc   --data_dir=../domainbed \
--algorithm ERM  --dataset ColoredMNIST  --trial_seed 0 --bias  0.6 0. --model_save \
--save_path stage1_${bias1} --epochs 1

CUDA_VISIBLE_DEVICES=3 python -m train_orth/train_disc  --data_dir=../domainbed \
--algorithm ERM  --dataset ColoredMNIST  --trial_seed 0 --bias  0.6 ${bias1} --model_save \
--save_path stage2_${bias1} --stage 2  --epochs 1

CUDA_VISIBLE_DEVICES=3 python -m train_orth/train_eval  --data_dir=../domainbed \
--algorithm ERM  --dataset ColoredMNIST --trial_seed 0 --bias  0.6 ${bias1} --save_path eval_${bias1}

## Running oracle
#CUDA_VISIBLE_DEVICES=3 python -m train_oracle   --data_dir=../domainbed \
#--algorithm ERM  --dataset ColoredMNIST  --trial_seed 0 --bias 0. 0.6 --epochs 1

#CUDA_VISIBLE_DEVICES=3 python -m train_orth  --data_dir=../domainbed --model_save \
#--algorithm ERM  --dataset ColoredMNIST --trial_seed 0 --bias  0.6 0.6 --save_path orth_0.6
#
#CUDA_VISIBLE_DEVICES=3 python -m eval_orth  --data_dir=../domainbed \
#--algorithm ERM  --dataset ColoredMNIST --trial_seed 0 --bias  0.6 0.6