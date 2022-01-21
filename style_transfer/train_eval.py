# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
from functools import partial
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from tqdm import tqdm
import dataset_domainbed
import hparams_registry
import algorithms
from util import misc
import torch.nn.functional as F

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
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--resume_path', default="model", type=str, help='path to resume the checkpoint')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--holdout_fraction', type=float, default=0.2)
parser.add_argument('--model_save', action='store_true')
parser.add_argument('--oc', action='store_true')
parser.add_argument('--re', action='store_true')
parser.add_argument('--oracle', action='store_true')
parser.add_argument('--bias', type=float, nargs='+', default=[0.6, 0])
parser.add_argument('--z1', type=float, default=0.6)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--irm_lam', type=float, default=1)
parser.add_argument('--rex_lam', type=float, default=1)
parser.add_argument('--cos_lam', type=float, default=1e-4)
parser.add_argument('--swap_lam', type=float, default=1.)
parser.add_argument('--trm_lam', type=float, default=1.)
parser.add_argument('--fish_lam', type=float, default=0.5)
parser.add_argument('--iters', type=int, default=1000)
parser.add_argument('--chunk', type=int, default=2)
parser.add_argument('--dro_eta', type=float, default=1e-2)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--class_balanced', action='store_true')
parser.add_argument('--age', action='store_true')
args = parser.parse_args()
args.step = 0
# If we ever want to implement checkpointing, just persist these values
# every once in a while, and then load them from disk here.
algorithm_dict = None
os.makedirs(args.output_dir, exist_ok=True)
sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

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
print("bias:", args.bias)
if args.dataset in vars(dataset_domainbed):
    dataset = vars(dataset_domainbed)[args.dataset](args.data_dir, args.bias, args.test_envs, hparams, eval=True)
else:
    raise NotImplementedError
fn = partial(os.path.join, '.', 'checkpoint', args.dataset)
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
train_loaders = [torch.utils.data.DataLoader(dataset=env, batch_size=hparams['batch_size'], shuffle=True,
                                             num_workers=dataset.N_WORKERS) for i, (env) in enumerate(in_splits) if
                 i not in args.test_envs]
eval_loaders = [torch.utils.data.DataLoader(dataset=env, batch_size=hparams['batch_size'], shuffle=False,
                                             num_workers=dataset.N_WORKERS)
    for env in (out_splits)]

eval_weights = [None for _ in (in_splits + out_splits)]
eval_loader_names = ['env{}_in'.format(i) for i in range(len(in_splits))]
eval_loader_names += ['env{}_out'.format(i) for i in range(len(out_splits))]

algorithm_class = algorithms.get_algorithm_class(args.algorithm)

if args.oc:
    print("Orthogonal Classifier:")
    if args.algorithm == 'ERM':
        path_inv = fn(f'stage1_{args.z1}_0_0')
    else:
        path_inv = fn(f'stage1_{args.z1}_{args.algorithm}')
    print("Resuming...", path_inv)
    checkpoint = torch.load(path_inv)
    algorithm_inv = algorithm_class(dataset.input_shape, dataset.num_classes,
                                    len(dataset) - len(args.test_envs), hparams)
    algorithm_inv.load_state_dict(checkpoint['model_dict'], strict=False)
    algorithm_inv.to(device)
    algorithm_inv.eval()

    path_full = fn(f'stage2_{args.z1}_{args.bias[1]}_{args.bias[2]}')
    print("Resuming...", path_full)
    algorithm_class = algorithms.get_algorithm_class('ERM')
    algorithm_full = algorithm_class(dataset.input_shape, dataset.num_classes,
                                     len(dataset) - len(args.test_envs), hparams)
    checkpoint = torch.load(path_full)
    algorithm_full.load_state_dict(checkpoint['model_dict'])
    algorithm_full.to(device)
    algorithm_full.eval()
elif args.re:
    print("Re:")
    print("Resuming...", fn(f'stage2_{args.z1}_{args.bias[1]}_{args.bias[2]}_reweight'))
    checkpoint = torch.load(fn(f'stage2_{args.z1}_{args.bias[1]}_{args.bias[2]}_reweight'))
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                    len(dataset) - len(args.test_envs), hparams)
    algorithm.load_state_dict(checkpoint['model_dict'])
    algorithm.to(device)
    algorithm.eval()
elif args.oracle:
    print("Oracle:")
    path_oracle = fn(f'stage3_0_{args.bias[1]}_{args.bias[2]}')
    print("Resuming...", path_oracle)
    checkpoint = torch.load(path_oracle)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                    len(dataset) - len(args.test_envs), hparams)
    algorithm.load_state_dict(checkpoint['model_dict'])
    algorithm.to(device)
    algorithm.eval()
else:
    raise NotImplementedError

steps_per_epoch = min([len(env) / hparams['batch_size'] for env in in_splits])
print("steps per epoch:", steps_per_epoch)
checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
last_results_keys = None
best_acc_in = 0.
best_acc_out = 0.

print("Eval: Evaluating orthogonal classifier")

def kl(pred, y):
    label = torch.zeros(len(y))
    prob = (1-args.bias[1])/2 + args.bias[1]
    label[y==0] = prob
    label[y==1] = 1-prob
    label = label.cuda()

    return - (label * torch.log(pred + 1e-3) + (1-label) * torch.log(1-pred+1e-3)).sum()

def main(epoch):

    global last_results_keys
    global best_acc_out
    global best_acc_in

    checkpoint_vals = collections.defaultdict(lambda: [])
    train_minibatches_iterator = eval_loaders[0]
    data_list = []
    label_list = []
    cnt = 0.
    loss = 0.0
    correct = 0.
    for x, label, _, y, _ in tqdm(train_minibatches_iterator):
        step_start_time = time.time()

        x = x.to(device)
        y_ = y.to(device)
        label = label.to(device)
        if args.oc:
            p_inv = F.softmax(algorithm_inv.predict(x), dim=1)
            p_full = F.softmax(algorithm_full.predict(x), dim=1)
            pred = p_full / p_inv
            pred = pred / pred.sum(1, keepdim=True)
        elif args.re:
            pred = F.softmax(algorithm.predict(x), dim=1)
            sum = torch.isnan(pred).float().sum()
            if sum > 0:
                #print("Nan Detect!")
                pred = torch.ones((len(pred), 2)).cuda() * 0.5
        elif args.oracle:
            pred = F.softmax(algorithm.predict(x), dim=1)
        else:
            raise NotImplementedError

        correct += (pred.max(1)[1].eq(y_)).sum()
        loss += F.cross_entropy(torch.log(pred+1e-5), label, reduction='sum').item()
        cnt += len(label)

        data_list += list(pred[:, 1].detach().cpu().data)
        label_list += list(y.data.numpy())
        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        args.step += 1

    print("Avg loss:", loss / cnt, ", Acc: ", correct / cnt)
    if args.oracle:
        name = 'or'
    elif args.oc:
        name = 'oc'
    else:
        name = 're'
    np.save(os.path.join(args.output_dir, f"bias_{args.bias[0]}_{args.bias[1]}_{args.bias[2]}_{name}_data"), np.array(data_list))
    np.save(os.path.join(args.output_dir, f"bias_{args.bias[0]}_{args.bias[1]}_{args.bias[2]}_{name}_label"), np.array(label_list))
    print("The statistics are saved at {}".format(os.path.join(args.output_dir, f"bias_{args.bias[0]}_{args.bias[1]}_{args.bias[2]}_{name}_data")))
if __name__ == "__main__":
    print("Training envs len:", len(train_loaders))
    for epoch in range(1):
        main(epoch)

