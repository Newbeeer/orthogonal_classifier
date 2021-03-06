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

import dataset_domainbed
import hparams_registry
import algorithms
from util import misc
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Domain generalization')
parser.add_argument('--data_dir', type=str, default='/data/scratch/ylxu/domainbed')
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
parser.add_argument('--output_dir', type=str, default="./checkpoint")
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--resume_path', default="model", type=str, help='path to resume the checkpoint')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--holdout_fraction', type=float, default=0.2)
parser.add_argument('--model_save', action='store_true')
parser.add_argument('--bias', type=float, nargs='+', default=[0.9, 0.8, 0.8])
parser.add_argument('--stage', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--cos_lam', type=float, default=1e-4)
parser.add_argument('--trm_lam', type=float, default=1.)
parser.add_argument('--fish_lam', type=float, default=0.5)
parser.add_argument('--dro_eta', type=float, default=1e-2)
parser.add_argument('--iters', type=int, default=1000)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--class_balanced', action='store_true')
args = parser.parse_args()
args.step = 0

os.makedirs(os.path.join('style_transfer', 'checkpoint', args.dataset), exist_ok=True)
fn = partial(os.path.join, 'style_transfer', 'checkpoint', args.dataset)
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

is_dg = (args.algorithm != 'ERM' and args.stage == 1)
if is_dg:
    args.test_envs = [0]

if args.dataset == 'Celeba':
    if args.stage == 1:
        # For training the $w_1$ classifier (principal classifier)
        group_names = ['male_nonblond_refine', 'male_blond_refine', 'female_blond_refine', 'female_nonblond_refine']
        dataset = vars(dataset_domainbed)[args.dataset](args.data_dir, group_names, args.stage, args.image_size, dg=is_dg)
    elif args.stage == 2:
        # For training the ground-truth $w_2$ classifier (the real orthogonal classifier)
        group_names = ['male_nonblond_refine', 'female_nonblond_refine', 'female_blond_refine', 'male_blond_refine']
        dataset = vars(dataset_domainbed)[args.dataset](args.data_dir, group_names, args.stage, args.image_size, dg=is_dg)
    elif args.stage == 3:
        # For training the full classifier $w_x$
        group_names = ['male_nonblond_refine', 'female_blond_refine']
        dataset = vars(dataset_domainbed)[args.dataset](args.data_dir, group_names, args.stage, args.image_size)
    else:
        raise NotImplementedError
    print(f"Group names:{group_names}, dataset len:{len(dataset)}")

elif args.dataset == 'CMNIST':
    args.epochs = 1 if not is_dg else 5
    if args.stage == 1:
        args.bias = [0.9, 0., 0.] if not is_dg else [0.9, 0.8, 0.4]
    elif args.stage == 2:
        args.bias = [0., 0.8, 0.8]
    elif args.stage == 3:
        args.bias = [0.9, 0.8, 0.8]
    dataset = vars(dataset_domainbed)[args.dataset](args.data_dir, args.bias, dg=is_dg)
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
    checkpoint = torch.load(f'style_transfer/checkpoint/{args.resume_path}.pth')
    algorithm.load_state_dict(checkpoint['model_dict'],strict=False)

algorithm.to(device)
steps_per_epoch = min([len(env) / hparams['batch_size'] for env in in_splits])
print("steps per epoch:", steps_per_epoch)
checkpoint_freq = 200
last_results_keys = None
best_acc_out = 0.
train_loaders = [torch.utils.data.DataLoader(dataset=env, batch_size=hparams['batch_size'],
                 num_workers=dataset.N_WORKERS, shuffle=True) for i, (env) in enumerate(in_splits) if
                 i not in args.test_envs]
print("train loader length:", len(train_loaders))


def main(epoch):

    global last_results_keys
    global best_acc_out

    checkpoint_vals = collections.defaultdict(lambda: [])
    train_minibatches_iterator = zip(*train_loaders)

    for batch in train_minibatches_iterator:
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
                              for x, y, _, _, _ in batch]
        step_vals = algorithm.update(minibatches_device)

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

            # calculate the out-domain (test) acc
            val_acc = results['env{}_out_acc'.format(0)]

            # saving the checkpoint
            if best_acc_out < val_acc or (epoch + 1) % 5 == 0:
                best_acc_out = val_acc
                print("best acc:", best_acc_out)
                if args.model_save:
                    save_dict = {
                        "args": vars(args),
                        "model_input_shape": dataset.input_shape,
                        "model_num_classes": dataset.num_classes,
                        "model_num_domains": len(dataset) - len(args.test_envs),
                        "model_hparams": hparams,
                        "model_dict": algorithm.state_dict()
                    }
                    torch.save(save_dict, fn(f'stage_{args.stage}_{args.algorithm}_{args.dataset}'))
                    print("Save to:", fn(f'stage_{args.stage}_{args.algorithm}_{args.dataset}'))

        if args.step > 2000:
            break
    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')


if __name__ == "__main__":

    for epoch in range(args.epochs):
        main(epoch)
    print(f"Best test acc:{best_acc_out}")
