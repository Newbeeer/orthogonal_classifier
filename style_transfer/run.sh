#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "run.sh <bias1>"
  exit 1
fi
bias0=$1
bias1=$2
bias2=$3
seed=$4
gpu=$5

CUDA_VISIBLE_DEVICES=${gpu} python -m train_w1_dg   --data_dir=~/domainbed \
--algorithm Fish  --dataset ColoredMNIST  --trial_seed ${seed} --bias ${bias0} 0.8 0.8 0.6 0.6 --model_save \
--save_path stage1_${bias0}_Fish --stage 1 --epochs 3

#CUDA_VISIBLE_DEVICES=${gpu} python -m train_w1_dg   --data_dir=~/domainbed \
#--algorithm ERM  --dataset ColoredMNIST  --trial_seed ${seed} --bias ${bias0} 0. 0. --model_save \
#--save_path stage1_${bias0}_0_0 --stage 1 --epochs 3
###
#CUDA_VISIBLE_DEVICES=${gpu} python -m train_w1_dg  --data_dir=../domainbed \
#--algorithm ERM  --dataset ColoredMNIST  --trial_seed ${seed} --bias  ${bias0} ${bias1} ${bias2} --model_save \
#--save_path stage2_${bias0}_${bias1}_${bias2} --stage 2  --epochs 3

CUDA_VISIBLE_DEVICES=${gpu} python -m train_eval  --data_dir=../domainbed \
--algorithm Fish  --dataset ColoredMNIST --trial_seed ${seed} --bias  0 ${bias1} ${bias2}  --z1 ${bias0} --oc
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m train_w1_dg   --data_dir=~/domainbed \
#--algorithm ERM  --dataset ColoredMNIST  --trial_seed ${seed} --bias 0. ${bias1} ${bias2} --model_save \
#--save_path stage3_0_${bias1}_${bias2}  --stage 3 --epochs 3
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m train_disc_weight_p  --data_dir=~/domainbed \
#--algorithm ERM  --dataset ColoredMNIST  --trial_seed ${seed} --bias ${bias0} ${bias1} ${bias2} --model_save \
#--resume_path checkpoint/ColoredMNIST/stage1_${bias0}_0_0 --save_path stage2_${bias0}_${bias1}_${bias2}_reweight --epochs 3

CUDA_VISIBLE_DEVICES=3 python -m train_disc_weight_p  --data_dir=~/domainbed \
--algorithm ERM  --dataset Celeba  --trial_seed 1  --model_save --stage 1


#CUDA_VISIBLE_DEVICES=1 python -m train_eval  --data_dir=../domainbed \
#--algorithm ERM  --dataset ColoredMNIST --trial_seed ${seed} --bias  0 0.6 0.8  --oracle

#CUDA_VISIBLE_DEVICES=${gpu} python -m train_eval  --data_dir=../domainbed \
#--algorithm ERM  --dataset ColoredMNIST --trial_seed ${seed} --bias  0 ${bias1} ${bias2}  --z1 ${bias0} --oc

#CUDA_VISIBLE_DEVICES=${gpu} python -m train_eval  --data_dir=../domainbed \
#--algorithm ERM  --dataset ColoredMNIST --trial_seed ${seed} --bias  0 0.6 0.8  --z1 ${bias0} --re

