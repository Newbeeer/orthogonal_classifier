set -ex
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --pool_size 50 --no_dropout \
--gan_mode vanilla --name vanilla


CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./datasets/cmnist_0.6_bkgd0 --name cmnist_cyclegan --model cycle_gan --pool_size 50 --no_dropout \
--gan_mode vanilla --name vanilla


CUDA_VISIBLE_DEVICES=0 python train_reweight.py --dataroot ./datasets/cmnist_0.6  --model cycle_gan --pool_size 50 --no_dropout \
--gan_mode vanilla --name reweight  --reweight --bias 0.6


CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./datasets/cmnist_0.6  --model cycle_gan --pool_size 50 --no_dropout \
--gan_mode vanilla --name vanilla --eval


CUDA_VISIBLE_DEVICES=1 python train_reweight_new.py --dataroot ./datasets/cmnist_0.6  \
--model cycle_gan --pool_size 50 --no_dropout \
--gan_mode vanilla --name mnist_js --eval --netG mnist \
--netD mnist --dataset ColoredMNIST --bias 0.6 0.6 0.6 --batch_size 128 --obj js --n_epochs 20

CUDA_VISIBLE_DEVICES=2 python train_reweight_new.py --dataroot ./datasets/cmnist_0.6  \
--model cycle_gan --pool_size 50 --no_dropout \
--gan_mode vanilla --name mnist_vanilla --eval --netG mnist \
--netD mnist --dataset ColoredMNIST --bias 1 0.6 0.6 --batch_size 128 --obj vanilla --n_epochs 12

CUDA_VISIBLE_DEVICES=1 python train_reweight_new.py --dataroot ./datasets/cmnist_0.6  \
--model cycle_gan --pool_size 50 --no_dropout \
--gan_mode vanilla --name vanilla --eval --netG celeba \
--netD mnist --dataset Celeba  --batch_size 128 --obj vanilla

CUDA_VISIBLE_DEVICES=1 python train_reweight_new.py --dataroot ./datasets/cmnist_0.6  \
--model cycle_gan --pool_size 50 --no_dropout \
--gan_mode vanilla --name vanilla --eval --netG resnet_6blocks \
--netD celeba --dataset Celeba  --batch_size 128 --obj vanilla


CUDA_VISIBLE_DEVICES=1 python train_reweight_new.py --dataroot ./datasets/cmnist_0.6  \
--model cycle_gan --pool_size 50 --no_dropout \
--gan_mode vanilla  --eval --netG unet_128 --netD basic \
--dataset Celeba  --batch_size 32 --obj kl --print_freq 5000 --name kl_bs32_pretrain --n_epochs 12 --image_size 128 --pretrain

CUDA_VISIBLE_DEVICES=3 python train_reweight_new.py --dataroot ./datasets/cmnist_0.6  \
--model cycle_gan --pool_size 50 --no_dropout \
--gan_mode vanilla  --eval --netG unet_128 --netD basic \
--dataset Celeba  --batch_size 32 --obj js --print_freq 5000 --name js_reweight_pretrain \
--n_epochs 12 --image_size 128 --pretrain --reweight

# 64
CUDA_VISIBLE_DEVICES=1 python train_reweight_new.py --dataroot ./datasets/cmnist_0.6  \
--model cycle_gan --pool_size 50 --no_dropout \
--gan_mode vanilla --name vanilla --eval --netG resnet_6blocks \
--netD celeba --dataset Celeba  --batch_size 100 --obj js --print_freq 5000 --name js_bs100_64 --n_epochs 12


CUDA_VISIBLE_DEVICES=2 python train_reweight_new.py --dataroot ./datasets/cmnist_0.6  \
--model cycle_gan --pool_size 50 --no_dropout \
--gan_mode vanilla --name vanilla --eval --netG unet_128 \
--netD basic --dataset Celeba  --batch_size 32 --obj vanilla --print_freq 5000 --name vanilla_bs32 --n_epochs 12 --image_size 128

# 64
CUDA_VISIBLE_DEVICES=2 python train_reweight_new.py --dataroot ./datasets/cmnist_0.6  \
--model cycle_gan --pool_size 50 --no_dropout \
--gan_mode vanilla --name vanilla --eval --netG resnet_6blocks \
--netD celeba --dataset Celeba  --batch_size 100 --obj vanilla --print_freq 5000 --name vanilla_bs100_64 --n_epochs 12 --display_port 8098

CUDA_VISIBLE_DEVICES=3 python train_reweight_new.py --dataroot ./datasets/cmnist_0.6  \
--model cycle_gan --pool_size 50 --no_dropout \
--gan_mode vanilla --name vanilla --eval --netG unet_128 \
--netD basic --dataset Celeba  --batch_size 32 --obj js --print_freq 5000 --name oracle_bs32 --n_epochs 12 --display_port 7777 --image_size 128