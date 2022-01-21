#!/bin/bash

#epoch=12
#name=mnist_vanilla
#obj=vanilla

#epoch=12
#name=vanilla_bs32
#obj=vanilla

#epoch=4
#name=js_bs32_pretrain
#obj=js


epoch=12
name=vanilla_gender_bs32
obj=vanilla
gpu=2
alg=ERM
#CUDA_VISIBLE_DEVICES=2 python generate.py --dataroot ./datasets/cmnist_0.6  \
#--model cycle_gan --pool_size 50 --no_dropout \
#--gan_mode vanilla --eval --netG unet_128 \
#--netD basic --dataset Celeba  --batch_size 32  --out_path celeba_${obj}_e${epoch} --obj ${obj} --name ${name} --resume_epoch ${epoch}
#
#
#python3 evaluation_tensorflow.py generated_images/celeba_${obj}_e${epoch} --name refine


CUDA_VISIBLE_DEVICES=${gpu} python generate.py --dataroot ./datasets/cmnist_0.6  \
--model cycle_gan --pool_size 50 --no_dropout \
--gan_mode vanilla --eval --netG unet_128 \
--netD basic --dataset Celeba  --batch_size 32 \
--image_size 128 --out_path ${name}_${epoch} --obj ${obj} --name ${name} \
--resume_epoch ${epoch} --alg ${alg} --gender

# CUDA_VISIBLE_DEVICES=${gpu} python3 evaluation_tensorflow.py generated_images/${name}_${epoch} --name refine --celeba --gpu 0


#CUDA_VISIBLE_DEVICES=2 python generate.py --dataroot ./datasets/cmnist_0.6  \
#--model cycle_gan --pool_size 50 --no_dropout \
#--gan_mode vanilla --eval --netG mnist \
#--netD mnist --dataset ColoredMNIST  --batch_size 32 \
#--out_path mnist_${obj}_e${epoch} --obj ${obj} --name ${name} \
#--resume_epoch ${epoch} --save