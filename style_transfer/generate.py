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
    if opt.dataset == 'ColoredMNIST':
        hparams = {'input_shape': (3,28,28),
                   'num_classes': 2,
                   'opt': 'SGD',
                   'lr': 1e-1,
                   'weight_decay': 1e-4,
                   'sch_size':600,
                   'alg': opt.alg
        }
    elif opt.dataset == 'Celeba':
        hparams = {'input_shape': (3, 128, 128),
                   'num_classes': 2,
                   'opt': 'SGD',
                   'lr': 1e-1,
                   'weight_decay': 1e-4,
                   'sch_size':600
        }

    if opt.dataset == 'Celeba':
        if opt.gender:
            gnames = ['male_refine', 'female_refine']
        else:
            gnames = ['male_nonblond_refine', 'female_blond_refine']
        print("Group names:", gnames)
        dataset = vars(dataset_domainbed)[opt.dataset](gnames=gnames, image_size=128)
    elif opt.dataset in vars(dataset_domainbed):
        dataset = vars(dataset_domainbed)[opt.dataset](opt.data_dir, opt.bias, [], hparams, eval=True)
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
    fn = partial(os.path.join, '.', 'checkpoint', opt.dataset)

    algorithm_class = algorithms.get_algorithm_class('ERM')
    # loading checkpoints
    w_o = algorithm_class(hparams['input_shape'], hparams['num_classes'], 1, hparams)
    if opt.dataset == 'Celeba':
        checkpoint = torch.load(fn('stage2_128_ERM_oracle'))
    else:
        checkpoint = torch.load(f'checkpoint/ColoredMNIST/stage3_{0.6}_{0.6}')
    w_o.load_state_dict(checkpoint['model_dict'])
    w_o.cuda()
    w_o.eval()

    w_1_oracle = algorithm_class(hparams['input_shape'], hparams['num_classes'], 1, hparams)
    if opt.dataset == 'Celeba':
        path_oracle = 'stage1_128_ERM_oracle_13'
        print("loading w1 oracle from ", fn(path_oracle))
        checkpoint = torch.load(fn(path_oracle))
    else:
        checkpoint = torch.load(f'checkpoint/ColoredMNIST/stage1_{0.6}_{0.6}')
    w_1_oracle.load_state_dict(checkpoint['model_dict'], strict=False)
    w_1_oracle.cuda()
    w_1_oracle.eval()

    w_1_algorithm_class = algorithms.get_algorithm_class(opt.alg)
    w_1 = w_1_algorithm_class(hparams['input_shape'], hparams['num_classes'], 1, hparams)
    if opt.dataset == 'Celeba':
        path_1 = f'stage1_male_nonblond_refine_male_blond_refine_128_{opt.alg}'
        if opt.oracle:
            path_1 = path_oracle
        print("loading w1 from ", fn(path_1))
        checkpoint = torch.load(fn(path_1))
    else:
        if opt.alg == 'ERM':
            checkpoint = torch.load(f'checkpoint/ColoredMNIST/stage1_{0.6}_{0}_{0}')
        else:
            checkpoint = torch.load(f'checkpoint/ColoredMNIST/stage1_{0.6}_{opt.alg}')
    w_1.load_state_dict(checkpoint['model_dict'], strict=False)
    w_1.cuda()
    w_1.eval()

    w_x = algorithm_class(hparams['input_shape'], hparams['num_classes'], 1, hparams)
    if opt.dataset == 'Celeba':
        checkpoint = torch.load(fn('stage2_male_nonblond_refine_female_blond_refine_128'))
    else:
        checkpoint = torch.load(f'checkpoint/ColoredMNIST/stage2_{0.6}_{0.6}')
    w_x.load_state_dict(checkpoint['model_dict'])
    w_x.cuda()
    w_x.eval()


    if not os.path.exists(os.path.join('./generated_images', opt.out_path)):
        os.makedirs(os.path.join('./generated_images', opt.out_path), exist_ok=True)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    dataset_size = len(eval_loader.dataset)
    model.load_networks(epoch=opt.resume_epoch)
    model.reset_eval_stats()
    
    for epoch in range(1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        for data in tqdm(eval_loader):  # inner loop within one epoch

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            if opt.reverse:
                model.set_input(data, net_1=w_o, net_x=w_x, net_o=w_1_oracle, net_1_o=w_o,)
            else:
                model.set_input(data, net_1=w_1, net_x=w_x, net_o=w_o, net_1_o=w_1_oracle,)         # unpack data from dataset and apply preprocessing
            model.forward_eval()
            model.generate(save=opt.save)
            model.eval_stats()
            iter_data_time = time.time()
        model.eval_stats(print_stat=True)
        # model.inception()
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        print(f"Save {model.cnt} images")