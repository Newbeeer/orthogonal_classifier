#!/bin/bash
#if [[ $# -ne 1 ]]; then
#  echo "run.sh <bias1>"
#  exit 1
#fi
#wd_value=1e-4
#bias0=$1
#bias1=$2


CUDA_VISIBLE_DEVICES=1 python -m train_w1_dg  --data_dir=~/domainbed --algorithm IRM \
--dataset Celeba  --trial_seed 1 --model_save --epochs 10 --stage 1

CUDA_VISIBLE_DEVICES=1 python train_cyclegan.py --dataroot ./datasets/cmnist_0.6  \
--model cycle_gan --pool_size 50 --no_dropout \
--gan_mode vanilla  --eval --netG unet_128 --netD basic \
--dataset Celeba  --batch_size 32 --obj js --print_freq 500 --name irm --n_epochs 10 --image_size 128 --pretrain --alg IRM


# reweight

#CUDA_VISIBLE_DEVICES=2 python -m train_disc_weight_p  --data_dir=~/domainbed \
#--algorithm ERM  --dataset Celeba --trial_seed 1 --model_save --epochs 1 --stage 2