

CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src cifar --tgt stl  --seed 123 --dw 1e-1

CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src cifar --tgt stl  --seed 123 --dw 1e-1 --iw

#CUDA_VISIBLE_DEVICES=0 python3 vada_train.py --r 0.7 0.3 --src cifar --tgt stl  --seed 12 --dw 1e-1 --orth
#
#CUDA_VISIBLE_DEVICES=0 python3 vada_train.py --r 0.7 0.3 --src cifar --tgt stl  --seed 1234 --dw 1e-1 --orth

CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src cifar --tgt stl  --seed 123 --dw 1e-1 --orth