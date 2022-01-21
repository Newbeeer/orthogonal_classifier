#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "run.sh <bias1>"
  exit 1
fi
wd_value=1e-4
bias0=$1
bias1=$2

#CUDA_VISIBLE_DEVICES=1 python -m train_disc   --data_dir=~/domainbed \
#--algorithm ERM  --dataset ColoredMNIST  --trial_seed ${seed} --bias ${bias0} 0. --model_save \
#--save_path stage1_${bias0}_${bias1}_reweight_${seed} --epochs 1

CUDA_VISIBLE_DEVICES=1 python -m train_disc_weight_p  --data_dir=~/domainbed \
--algorithm ERM  --dataset ColoredMNIST  --trial_seed 0 --bias ${bias0} ${bias1} --model_save \
--resume_path checkpoint/ColoredMNIST/stage1_0.6_0.6 --save_path stage2_${bias0}_${bias1}_reweight --epochs 3

#CUDA_VISIBLE_DEVICES=1 python -m train_eval_weight_p  --data_dir=~/domainbed \
#--algorithm ERM  --dataset ColoredMNIST --trial_seed ${seed} --bias ${bias0} ${bias1} \
#--resume_path stage2_${bias0}_${bias1}_reweight_${seed}

#CUDA_VISIBLE_DEVICES=3 python -m train_eval_z1  --data_dir=~/domainbed \
#--algorithm ERM  --dataset ColoredMNIST --trial_seed 0 --bias 0.6 0.6
#
#CUDA_VISIBLE_DEVICES=3 python -m train_disc_weight_p  --data_dir=~/domainbed \
#--algorithm ERM  --dataset ColoredMNIST  --trial_seed 0 --bias 0.6 0.6 --model_save \
#--resume_path stage1_0.6 --save_path stage2_0.6_reweight --epochs 4