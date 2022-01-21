#CUDA_VISIBLE_DEVICES=2 python3 vada_train.py --r 1 1 --src mnist --tgt mnistm --orth --seed 1234 --batch_size 128
#
#CUDA_VISIBLE_DEVICES=2 python3 vada_train.py --r 1 1 --src mnist --tgt mnistm --seed 1234 --batch_size 128
#
#97.5,97.0 CUDA_VISIBLE_DEVICES=2 python3 vada_train.py --r 1 1 --src mnist --tgt mnistm --orth --seed 12 --batch_size 128
#
#94.7,94.1 CUDA_VISIBLE_DEVICES=2 python3 vada_train.py --r 1 1 --src mnist --tgt mnistm --seed 12 --batch_size 128

#CUDA_VISIBLE_DEVICES=2 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt mnistm --orth --seed 12
#
#CUDA_VISIBLE_DEVICES=0 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt mnistm --seed 12 --iw
#
#CUDA_VISIBLE_DEVICES=0 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt mnistm --seed 12 --source

#CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.5 0.5 --src mnist --tgt mnistm --seed 1234 --batch_size 256 --balance
#
#CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.5 0.5 --src mnist --tgt mnistm --orth --seed 12 --batch_size 256 --balance
#
#CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.5 0.5 --src mnist --tgt mnistm --seed 12 --batch_size 256 --balance

CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt mnistm  --seed 1 --orth

CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt mnistm  --seed 12 --orth

CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt mnistm  --seed 1234 --orth