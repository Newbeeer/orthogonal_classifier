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
from util.visualizer import Visualizer
import torch
import algorithms
from util import misc
import dataset_domainbed
if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    torch.manual_seed(opt.trial_seed)

    # prepare the dataset
    if opt.dataset == 'ColoredMNIST':
        hparams = {'input_shape': (3,28,28),
                   'num_classes': 2,
                   'opt': 'SGD',
                   'lr': 1e-1,
                   'weight_decay': 1e-4,
                   'sch_size': 600,
                   'alg': opt.alg
        }
    elif opt.dataset == 'Celeba':
        hparams = {'input_shape': (3, opt.image_size, opt.image_size),
                   'num_classes': 2,
                   'opt': 'SGD',
                   'lr': 1e-1,
                   'weight_decay': 1e-4,
                   'sch_size': 600
        }
    if not os.path.exists(os.path.join('.', opt.dataset)):
        os.makedirs(os.path.join('.', 'checkpoint', opt.dataset), exist_ok=True)
    fn = partial(os.path.join, '.', 'checkpoint', opt.dataset)
    if opt.dataset == 'Celeba':
        # gnames = ['young_blond', 'old_nonblond']
        if opt.gender:
            gnames = ['male_refine', 'female_refine']
        else:
            gnames = ['male_nonblond_refine', 'female_nonblond_refine']
        print("Group names:", gnames)
        dataset = vars(dataset_domainbed)[opt.dataset](gnames=gnames, image_size=opt.image_size)
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

    for i, (env) in enumerate(in_splits):
        print("Env:{}, dataset len:{}".format(i, len(env)))

    # class of env: ImageFolder
    train_loader = torch.utils.data.DataLoader(dataset=in_splits[0], batch_size=opt.batch_size, shuffle=True,
                                                 num_workers=dataset.N_WORKERS)
    eval_loader = torch.utils.data.DataLoader(dataset=out_splits[0], batch_size=opt.batch_size, shuffle=False,
                                                 num_workers=dataset.N_WORKERS)

    algorithm_class = algorithms.get_algorithm_class('ERM')
    # loading checkpoints
    w_o = algorithm_class(hparams['input_shape'], hparams['num_classes'], 1, hparams)
    if opt.dataset == 'Celeba':
        path = fn('stage2_128_ERM_oracle')
        print("loading w_o oracle from ", (path))
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(f'checkpoint/ColoredMNIST/stage3_{0}_{0.8}_{0.8}')
    w_o.load_state_dict(checkpoint['model_dict'])
    w_o.cuda()
    w_o.eval()

    w_1_oracle = algorithm_class(hparams['input_shape'], hparams['num_classes'], 1, hparams)
    if opt.dataset == 'Celeba':
        path_oracle = 'stage1_128_ERM_oracle_13'
        print("loading w1 oracle from ", fn(path_oracle))
        checkpoint = torch.load(fn(path_oracle))
    else:
        checkpoint = torch.load(f'checkpoint/ColoredMNIST/stage1_{0.9}_{0}_{0}')
    w_1_oracle.load_state_dict(checkpoint['model_dict'], strict=False)
    w_1_oracle.cuda()
    w_1_oracle.eval()

    w_1_algorithm_class = algorithms.get_algorithm_class(opt.alg)
    w_1 = w_1_algorithm_class(hparams['input_shape'], hparams['num_classes'], 1, hparams)
    if opt.dataset == 'Celeba':
        path_1 = f'stage1_male_nonblond_refine_male_blond_refine_128_{opt.alg}'
        if opt.alg == 'Fish':
            path_1 = f'stage1_male_nonblond_refine_male_blond_refine_128_{opt.alg}_{800}'
        elif opt.alg == 'MLDG':
            path_1 = f'stage1_male_nonblond_refine_male_blond_refine_128_{opt.alg}_{1500}'
        if opt.gender:
            # MLDG - 4100, Fish
            # path_1 = f'stage1_male_young_refine_female_young_refine_128_{opt.alg}_{4900}'
            # path_1 = f'stage1_male_young_refine_female_young_refine_128_{opt.alg}'
            pass
        if opt.oracle:
            path_1 = path_oracle
        print("loading w1 from ", fn(path_1))
        checkpoint = torch.load(fn(path_1))
    else:
        if opt.alg == 'ERM':
            path_1 = f'checkpoint/ColoredMNIST/stage1_{0.9}_{0}_{0}'
        else:
            path_1 = f'checkpoint/ColoredMNIST/stage1_{0.9}_{opt.alg}'
        print("loading w1 from ", (path_1))
        checkpoint = torch.load(path_1)
    w_1.load_state_dict(checkpoint['model_dict'], strict=False)
    w_1.cuda()
    w_1.eval()


    w_x = algorithm_class(hparams['input_shape'], hparams['num_classes'], 1, hparams)
    if opt.dataset == 'Celeba':
        if opt.gender:
            #path = 'stage3_128_ERM_oracle_0'
            path = fn(f'stage1_male_young_refine_female_young_refine_128_ERM_{4000}')
        else:
            path = fn('stage2_male_nonblond_refine_female_blond_refine_128')
        print("Loading w_x from ", path)
        checkpoint = torch.load(path)
    else:
        path = f'checkpoint/ColoredMNIST/stage2_{0.9}_{0.8}_{0.8}'
        print("Loading w_x from ", path)
        checkpoint = torch.load(path)
    w_x.load_state_dict(checkpoint['model_dict'])
    w_x.cuda()
    w_x.eval()


    if opt.reweight:
        algorithm_reweight = algorithm_class(hparams['input_shape'], hparams['num_classes'], 1, hparams)
        if opt.dataset == 'Celeba':
            checkpoint = torch.load(fn('reweight_male_nonblond_refine_female_blond_refine_128'))
        else:
            checkpoint = torch.load(f'checkpoint/ColoredMNIST/stage2_0.9_{0.8}_{0.8}_reweight')
        algorithm_reweight.load_state_dict(checkpoint['model_dict'])
        algorithm_reweight.cuda()
        algorithm_reweight.eval()
    else:
        algorithm_reweight = None


    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    dataset_size = len(train_loader.dataset)

    if opt.pretrain:
        # loading pretrained cyclegan (12 epochs)
        if opt.gender:
            model.load_networks(epoch=12, pre_name='vanilla_gender_bs32', pre_obj='vanilla')
        else:
            if opt.dataset == 'ColoredMNIST':
                model.load_networks(epoch=4, pre_name='mnist_vanilla', pre_obj='vanilla_ERM')
            else:
                model.load_networks(epoch=12, pre_name='vanilla_bs32', pre_obj='vanilla')
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        model.reset_eval_stats()
        for i, data in enumerate(train_loader):  # inner loop within one epoch

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += 1
            epoch_iter += 1
            if opt.reverse:
                model.set_input(data, net_1=w_o, net_x=w_x, net_o=w_1_oracle, net_1_o=w_o,
                                net_reweight=algorithm_reweight)
            else:
                model.set_input(data, net_1=w_1, net_x=w_x, net_o=w_o, net_1_o=w_1_oracle,
                                net_reweight=algorithm_reweight)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            model.eval_stats()
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                if opt.eval:
                    model.eval_stats(print_stat=True)
        model.reset_eval_stats()
        for i, data in enumerate(eval_loader):  # inner loop within one epoch
            if opt.reverse:
                model.set_input(data, net_1=w_o, net_x=w_x, net_o=w_1_oracle, net_1_o=w_o,
                                net_reweight=algorithm_reweight)
            else:
                model.set_input(data, net_1=w_1, net_x=w_x, net_o=w_o, net_1_o=w_1_oracle,
                                net_reweight=algorithm_reweight)  # unpack data from dataset and apply preprocessing
            model.forward_eval()
            model.eval_stats()
            iter_data_time = time.time()
        model.eval_stats(print_stat=True, eval=True)
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
