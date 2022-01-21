#CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn --orth --seed 1234

#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn --seed 1 --dann

#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.3 1 --src svhn --tgt mnist --orth --seed 1 --dann
#
#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.3 1 --src svhn --tgt mnist --seed 1 --dann


CUDA_VISIBLE_DEVICES=2 python3 vada_train.py --r 0.7 0.3 --src svhn --tgt mnist --seed 1234 --orth

CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src svhn --tgt mnist --seed 123 --orth


CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 1 1 --src svhn --tgt mnist --seed 123 --adda --pretrain_epoch 20

CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 1 1 --src svhn --tgt mnist --seed 123 --adda --pretrain_epoch 20

CUDA_VISIBLE_DEVICES=4 python3 vada_train.py --r 0.7 0.3 --src svhn2mnist --tgt mnist --seed 123 --adda --resume --resume_epoch 10 --pretrain_epoch 0 --orth


CUDA_VISIBLE_DEVICES=5 python3 vada_train.py --r 0.7 0.3 --src stl --tgt cifar --seed 123 --adda  --pretrain_epoch 10