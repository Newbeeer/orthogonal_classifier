# 0.1 fails

#CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn --orth --seed 1234

#CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn --seed 1 --dann



CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src mnist2svhn --tgt svhn  --seed 123 --adda --pretrain_epoch 10

CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn  --seed 12 --orth

CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn  --seed 123 --orth
#CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn --seed 12 --dann

#CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn --orth --seed 12
#
#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn --iw --seed 12
#
#CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 1 1 --src mnist --tgt svhn --seed 12

#CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn --orth --seed 123
#
#CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn --orth --seed 123

