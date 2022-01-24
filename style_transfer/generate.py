"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
from functools import partial
import os
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import torch
from tqdm import tqdm
from util import misc
import dataset_domainbed
from torchvision import transforms
from datasets.celeba_dataset import CelebA_group
import algorithms

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    opt.generate = True
    torch.manual_seed(123)
    # prepare the dataset
    if opt.dataset == 'CMNIST':
        opt.batch_size = 128
        opt.pretrain_epoch = 0
        opt.netG = 'mnist'
        opt.netD = 'mnist'
        hparams = {'input_shape': (3,28,28),
                   'num_classes': 2,
                   'opt': 'SGD',
                   'lr': 1e-1,
                   'weight_decay': 1e-4,
                   'sch_size': 600,
                   'alg': opt.alg
        }
    elif opt.dataset == 'Celeba':
        opt.batch_size = 32
        opt.pretrain_epoch = 12
        opt.netG = 'unet_128'
        opt.netD = 'basic'
        hparams = {'input_shape': (3, 128, 128),
                   'num_classes': 2,
                   'opt': 'SGD',
                   'lr': 1e-1,
                   'weight_decay': 1e-4,
                   'sch_size': 600
        }

    if not os.path.exists(os.path.join('.', opt.dataset)):
        os.makedirs(os.path.join('style_transfer', 'checkpoint', opt.dataset), exist_ok=True)
    fn = partial(os.path.join, 'style_transfer', 'checkpoint', opt.dataset)
    if opt.dataset == 'Celeba':
        if opt.gender:
            group_names = ['male_refine', 'female_refine']
        else:
            group_names = ['male_nonblond_refine', 'female_nonblond_refine']
        print("Group names:", group_names)
        dataset = vars(dataset_domainbed)[opt.dataset](opt.data_dir, group_names, stage=3)
    elif opt.dataset in vars(dataset_domainbed):
        dataset = vars(dataset_domainbed)[opt.dataset](opt.data_dir, opt.bias)
    else:
        raise NotImplementedError


    in_splits = []
    out_splits = []
    opt.holdout_fraction = 0.2

    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(env, int(len(env) * opt.holdout_fraction),
                                      misc.seed_hash(opt.trial_seed, env_i))
        in_weights, out_weights = None, None
        in_splits.append(in_)
        out_splits.append(out)

    # class of env: ImageFolder
    torch.manual_seed(123)

    eval_loader = torch.utils.data.DataLoader(dataset=out_splits[0], batch_size=64, shuffle=False,
                                                 num_workers=dataset.N_WORKERS)
    print(f"Len eval:{len(eval_loader.dataset)}")
    fn = partial(os.path.join, 'style_transfer', 'checkpoint', opt.dataset)

    algorithm_class = algorithms.get_algorithm_class('ERM')
    # loading checkpoints
    # ground-truth w_2 orthogonal classifier
    w_2 = algorithm_class(hparams['input_shape'], hparams['num_classes'], 1, hparams)
    path = fn(f'stage_2_ERM_{opt.dataset}')
    print("loading ground truth w_2 oracle from ", path)
    checkpoint = torch.load(path)
    w_2.load_state_dict(checkpoint['model_dict'])
    w_2.cuda()
    w_2.eval()

    w_1_oracle = algorithm_class(hparams['input_shape'], hparams['num_classes'], 1, hparams)
    path_oracle = fn(f'stage_1_ERM_{opt.dataset}')
    print("loading w_1 oracle from ", path_oracle)
    checkpoint = torch.load(path_oracle)
    w_1_oracle.load_state_dict(checkpoint['model_dict'], strict=False)
    w_1_oracle.cuda()
    w_1_oracle.eval()

    w_x = algorithm_class(hparams['input_shape'], hparams['num_classes'], 1, hparams)
    path = fn(f'stage_3_ERM_{opt.dataset}')
    print("loading w_x from ", path)
    checkpoint = torch.load(path)
    w_x.load_state_dict(checkpoint['model_dict'])
    w_x.cuda()
    w_x.eval()


    os.makedirs(os.path.join('style_transfer/generated_images', opt.out_path), exist_ok=True)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    dataset_size = len(eval_loader.dataset)
    model.load_networks(epoch=opt.resume_epoch, pre_name=opt.name, pre_obj='orthogonal')
    model.reset_eval_stats()
    
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    for data in tqdm(eval_loader):  # inner loop within one epoch

        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data, epoch=-1, net_1=w_1_oracle, net_x=w_x, net_2=w_2, net_1_o=w_1_oracle,)
        model.forward_eval()

        model.generate(save=opt.save)
        model.eval_stats()
        iter_data_time = time.time()
    model.eval_stats(print_stat=True)
    print(model.cnt_img.values())
    print(f"Save {sum(model.cnt_img.values())} images")