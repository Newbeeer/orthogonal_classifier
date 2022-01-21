#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "run.sh <bias1>"
  exit 1
fi

alg=$1
seed=$2
gpu=$3
bias0=$4
bias1=0.8
bias2=0.8

#CUDA_VISIBLE_DEVICES=${gpu} python -m train_w1_dg   --data_dir=~/domainbed \
#--algorithm ${alg}  --dataset ColoredMNIST  --trial_seed ${seed} --bias ${bias0} 0.5 0.7 0.7 0.9 --model_save \
#--save_path stage1_${bias0}_${alg} --stage 1 --epochs 12

CUDA_VISIBLE_DEVICES=${gpu} python -m train_w1_dg   --data_dir=~/domainbed \
--algorithm ERM  --dataset ColoredMNIST  --trial_seed ${seed} --bias ${bias0} 0. 0. --model_save \
--save_path stage1_${bias0}_0_0 --stage 1 --epochs 1

#CUDA_VISIBLE_DEVICES=${gpu} python -m train_w1_dg  --data_dir=../domainbed \
#--algorithm ERM  --dataset ColoredMNIST  --trial_seed ${seed} --bias  ${bias0} ${bias1} ${bias2} --model_save \
#--save_path stage2_${bias0}_${bias1}_${bias2} --stage 2  --epochs 1

#CUDA_VISIBLE_DEVICES=3 python -m train_w1_dg   --data_dir=~/domainbed \
#--algorithm ERM  --dataset ColoredMNIST  --trial_seed ${seed} --bias 0. ${bias1} ${bias2} --model_save \
#--save_path stage3_0_${bias1}_${bias2}  --stage 3 --epochs 1

CUDA_VISIBLE_DEVICES=${gpu} python -m train_disc_weight_p  --data_dir=~/domainbed \
--algorithm ERM  --dataset ColoredMNIST  --trial_seed ${seed} --bias ${bias0} ${bias1} ${bias2} --model_save \
--resume_path checkpoint/ColoredMNIST/stage1_${bias0}_0_0 --save_path stage2_${bias0}_${bias1}_${bias2}_reweight \
--epochs 3

#CUDA_VISIBLE_DEVICES=1 python -m train_eval  --data_dir=../domainbed \
#--algorithm ERM  --dataset ColoredMNIST --trial_seed ${seed} --bias  0 0.8 0.8  --oracle

CUDA_VISIBLE_DEVICES=${gpu} python -m train_eval  --data_dir=../domainbed \
--algorithm ERM  --dataset ColoredMNIST --trial_seed ${seed} --bias  0 0.8 0.8  --z1 ${bias0} --re

#CUDA_VISIBLE_DEVICES=${gpu} python -m train_eval  --data_dir=../domainbed \
#--algorithm ${alg}  --dataset ColoredMNIST --trial_seed ${seed} --bias  0 0.8 0.8  --z1 ${bias0} --oc



