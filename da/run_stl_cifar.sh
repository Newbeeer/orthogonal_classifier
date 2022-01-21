#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src stl --tgt cifar  --seed 12
#
#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src stl --tgt cifar  --seed 1

#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src stl --tgt cifar  --seed 12 --dw 1e-1 --dann

#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src stl --tgt cifar  --seed 12 --dw 1e-1 --orth
#
#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src stl --tgt cifar  --seed 12 --dw 1e-1

#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src stl --tgt cifar  --seed 1 --dw 1e-1 --dann

#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src stl --tgt cifar  --seed 12 --dw 1e-2 --orth
#
#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src stl --tgt cifar  --seed 12 --dw 1e-2
#
#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src stl --tgt cifar  --seed 123 --dw 1e-2 --orth
#
#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src stl --tgt cifar  --seed 123 --dw 1e-2

CUDA_VISIBLE_DEVICES=2 python3 vada_train.py --r 0.7 0.3 --src stl --tgt cifar  --seed 123 --dw 1e-2 --orth

CUDA_VISIBLE_DEVICES=0 python3 vada_train.py --r 0.7 0.3 --src stl --tgt cifar  --seed 123 --dw 1e-2 --iw



