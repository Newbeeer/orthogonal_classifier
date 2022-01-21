#CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn --orth --seed 1234

#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src signs --tgt gtsrb --seed 12 --iw

CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src signs --tgt gtsrb --seed 123 --orth

CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src signs --tgt gtsrb --seed 12 --orth

# CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src signs --tgt gtsrb --seed 1234 --orth
