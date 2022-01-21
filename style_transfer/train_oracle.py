# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
from tqdm import tqdm
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

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
parser.add_argument('--select', action='store_true')
parser.add_argument('--descend', action='store_true')
parser.add_argument('--flip', action='store_true')
parser.add_argument('--bias', type=float, nargs='+', default=[0.6, 0.6, 0.6])
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

if args.dataset in vars(dataset_domainbed):
    dataset = vars(dataset_domainbed)[args.dataset](args.data_dir, args.bias, args.test_envs, hparams)
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
eval_loaders = [torch.utils.data.DataLoader(dataset=env, batch_size=hparams['batch_size'], shuffle=False,
                                             num_workers=dataset.N_WORKERS)
    for env in (in_splits + out_splits)]

eval_weights = [None for _ in (in_splits + out_splits)]
eval_loader_names = ['env{}_in'.format(i) for i in range(len(in_splits))]
eval_loader_names += ['env{}_out'.format(i) for i in range(len(out_splits))]

algorithm_class = algorithms.get_algorithm_class(args.algorithm)
algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                            len(dataset) - len(args.test_envs), hparams)
if algorithm_dict is not None:
    algorithm.load_state_dict(algorithm_dict)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{args.resume_path}.pth')
    algorithm.load_state_dict(checkpoint['model_dict'],strict=False)

algorithm.to(device)
steps_per_epoch = min([len(env) / hparams['batch_size'] for env in in_splits])
print("steps per epoch:", steps_per_epoch)
checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
last_results_keys = None
best_acc_in = 0.
best_acc_out = 0.


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
    train_loaders = torch.utils.data.DataLoader(dataset=in_splits[0], batch_size=hparams['batch_size'], shuffle=True,
                                                 num_workers=dataset.N_WORKERS)
    train_minibatches_iterator = train_loaders
    for batch in train_minibatches_iterator:
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
                              for x, y, _, _, _ in [batch]]
        step_vals = algorithm.update(minibatches_device,new_loss=0.)

        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        args.step += 1
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if args.step % checkpoint_freq == 0:
            results = {
                'step': args.step,
                'epoch': epoch,
            }
            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)
            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name + '_acc'] = acc

            results_keys = sorted(results.keys())
            misc.print_row(results_keys, colwidth=12)
            misc.print_row([results[key] for key in results_keys],
                           colwidth=12)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')


def eval():

    global last_results_keys
    global best_acc_out
    global best_acc_in
    train_loaders = torch.utils.data.DataLoader(dataset=in_splits[0], batch_size=hparams['batch_size'], shuffle=True,
                                                 num_workers=dataset.N_WORKERS)
    checkpoint_vals = collections.defaultdict(lambda: [])
    train_minibatches_iterator = train_loaders
    data_list = []
    label_list = []
    kl_sum = 0.
    cnt = 0.
    for x, _, _, y, _ in tqdm(train_minibatches_iterator):
        step_start_time = time.time()
        x = x.to(device)
        y_ = y.to(device)
        p_partial = F.softmax(algorithm.predict(x), dim=1)[:, 0]
        data_list += list(p_partial.cpu().data)
        label_list += list(y.data.numpy())
        kl_sum += kl(p_partial, y_).item()
        cnt += len(y)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        args.step += 1

    print("Avg kl:", kl_sum / cnt)
    np.save(os.path.join(args.output_dir, f"bias_{args.bias[1]}_data_oracle"), np.array(data_list))
    np.save(os.path.join(args.output_dir, f"bias_{args.bias[1]}_label_oracle"), np.array(label_list))
    print("The statistics are saved at {}".format(os.path.join(args.output_dir, f"bias_{args.bias[1]}_data_oracle")))


if __name__ == "__main__":

    for epoch in range(args.epochs):
        main(epoch)
    print(f"Best in acc:{best_acc_in}, Best out acc:{best_acc_out}")
    eval()