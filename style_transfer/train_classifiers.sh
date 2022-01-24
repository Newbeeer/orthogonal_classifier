#!/bin/bash

gpu=$1
root=$2
dataset=$3
alg=$4


# train the principal classifier
CUDA_VISIBLE_DEVICES=${gpu}  python style_transfer/train_classifier.py  --data_dir=${root} --algorithm ${alg} \
  --dataset ${dataset}  --model_save --epochs 10 --stage 1

# train the (ground-truth) orthogonal classifier (for z2 acc)
CUDA_VISIBLE_DEVICES=${gpu}  python  style_transfer/train_classifier.py  --data_dir=${root} --algorithm ${alg} \
  --dataset ${dataset}  --model_save --epochs 10 --stage 2

# train the full classifier
CUDA_VISIBLE_DEVICES=${gpu}  python style_transfer/train_classifier.py  --data_dir=${root} --algorithm ${alg} \
  --dataset ${dataset}  --model_save --epochs 1 --stage 3

