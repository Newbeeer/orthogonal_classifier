# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from PIL import Image
import datasets
import hparams_registry

from util import misc


parser = argparse.ArgumentParser(description='Domain generalization')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--dataset', type=str, default="RotatedMNIST")
parser.add_argument('--algorithm', type=str, default="ERM")
parser.add_argument('--opt', type=str, default="SGD")
parser.add_argument('--hparams', type=str,
                    help='JSON-serialized hparams dict')
parser.add_argument('--hparams_seed', type=int, default=0,
                    help='Seed for random hparams (0 means "default hparams")')
parser.add_argument('--trial_seed', type=int, default=0,
                    help='Trial number (used for seeding split_dataset and '
                         'random_hparams).')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed for everything else')
parser.add_argument('--epochs', type=int, default=10,
                    help='Training epochs')
parser.add_argument('--steps', type=int, default=None,
                    help='Number of steps. Default is dataset-dependent.')
parser.add_argument('--checkpoint_freq', type=int, default=None,
                    help='Checkpoint every N steps. Default is dataset-dependent.')
parser.add_argument('--test_envs', type=int, nargs='+', default=[])
parser.add_argument('--output_dir', type=str, default="checkpoint")
parser.add_argument('--save_path', type=str, default="model")
parser.add_argument('--suffix', type=str, default="")
parser.add_argument('--resume_path', default="model", type=str, help='path to resume the checkpoint')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--holdout_fraction', type=float, default=0.2)
parser.add_argument('--bias', type=float, default=0.9)
parser.add_argument('--skip_model_save', action='store_true')
parser.add_argument('--select', action='store_true')
parser.add_argument('--descend', action='store_true')
parser.add_argument('--select_fractions', type=float, default=0.99, help='The fractions of selected data')
args = parser.parse_args()
args.step = 0
# If we ever want to implement checkpointing, just persist these values
# every once in a while, and then load them from disk here.
algorithm_dict = None

os.makedirs(args.output_dir, exist_ok=True)
sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

print("Environment:")
print("\tPython: {}".format(sys.version.split(" ")[0]))
print("\tPyTorch: {}".format(torch.__version__))
print("\tTorchvision: {}".format(torchvision.__version__))
print("\tCUDA: {}".format(torch.version.cuda))
print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
print("\tNumPy: {}".format(np.__version__))
print("\tPIL: {}".format(PIL.__version__))

print('Args:')
for k, v in sorted(vars(args).items()):
    print('\t{}: {}'.format(k, v))

if args.hparams_seed == 0:
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset, args)
else:
    hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                              misc.seed_hash(args.hparams_seed, args.trial_seed))
if args.hparams:
    hparams.update(json.loads(args.hparams))

print('HParams:')
for k, v in sorted(hparams.items()):
    print('\t{}: {}'.format(k, v))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if args.dataset in vars(datasets):
    dataset = vars(datasets)[args.dataset](args.data_dir, args.bias, args.test_envs, hparams)
else:
    raise NotImplementedError

# Split each env into an 'in-split' and an 'out-split'. We'll train on
# each in-split except the test envs, and evaluate on all splits.
in_splits = []
out_splits = []
for env_i, env in enumerate(dataset):
    out, in_ = misc.split_dataset(env, int(len(env) * args.holdout_fraction), misc.seed_hash(args.trial_seed, env_i))
    if hparams['class_balanced']:
        in_weights = misc.make_weights_for_balanced_classes(in_)
        out_weights = misc.make_weights_for_balanced_classes(out)
    else:
        in_weights, out_weights = None, None
    in_splits.append(in_)
    out_splits.append(out)

for i, (env) in enumerate(in_splits):
    print("Env:{}, dataset len:{}".format(i, len(env)))

# class of env: ImageFolder
train_loaders = [torch.utils.data.DataLoader(dataset=env, batch_size=1, shuffle=True,
                                             num_workers=dataset.N_WORKERS) for i, (env) in enumerate(in_splits)]
test_loaders = [torch.utils.data.DataLoader(dataset=env, batch_size=1, shuffle=False,
                                             num_workers=dataset.N_WORKERS) for env in (out_splits) ]
map_dict = {0: 'A', 1: 'B'}

import os
from tqdm import tqdm
os.makedirs(f"./datasets/cmnist_{args.bias}_{args.suffix}/trainA",exist_ok=True)
os.makedirs(f"./datasets/cmnist_{args.bias}_{args.suffix}/trainB",exist_ok=True)
os.makedirs(f"./datasets/cmnist_{args.bias}_{args.suffix}/testA",exist_ok=True)
os.makedirs(f"./datasets/cmnist_{args.bias}_{args.suffix}/testB",exist_ok=True)
def main():

    pos = 0
    neg = 0
    for (img, label) in (tqdm(train_loaders[0])):
        img = img.numpy()[0].transpose(1,2,0)
        label = int(label[0])
        #print("Image shape:,", img.shape, np.max(img))
        im = Image.fromarray((img * 255).astype(np.uint8))

        idx = neg if label == 0 else pos
        im.save(f"./datasets/cmnist_{args.bias}_{args.suffix}/train{map_dict[label]}/{idx}_{map_dict[label]}.jpg")
        if label == 0:
            neg += 1
        else:
            pos += 1
    print(f"Train : pos :{pos}, neg :{neg}")
    pos = 0
    neg = 0
    for (img, label) in (tqdm(test_loaders[0])):
        img = img.numpy()[0].transpose(1,2,0)
        label = int(label[0])
        #print("Image shape:,", img.shape, np.max(img))
        im = Image.fromarray((img * 255).astype(np.uint8))
        idx = neg if label == 0 else pos
        im.save(f"./datasets/cmnist_{args.bias}_{args.suffix}/test{map_dict[label]}/{idx}_{map_dict[label]}.jpg")
        if label == 0:
            neg += 1
        else:
            pos += 1

    print(f"Test : pos :{pos}, neg :{neg}")



if __name__ == "__main__":

    main()
