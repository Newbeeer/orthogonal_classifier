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
    if opt.dataset == 'CMNIST':
        opt.batch_size = 128
        opt.pretrain_epoch = 0
        opt.n_epochs = opt.pretrain_epoch + 6
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
        opt.n_epochs = opt.pretrain_epoch + 12
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

    for i, (env) in enumerate(in_splits):
        print("Env:{}, dataset len:{}".format(i, len(env)))

    # class of env: ImageFolder
    train_loader = torch.utils.data.DataLoader(dataset=in_splits[0], batch_size=opt.batch_size, shuffle=True,
                                                 num_workers=dataset.N_WORKERS)
    eval_loader = torch.utils.data.DataLoader(dataset=out_splits[0], batch_size=opt.batch_size, shuffle=False,
                                                 num_workers=dataset.N_WORKERS)

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


    def train(model, epoch):

        global total_iters
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.reset_eval_stats()
        for i, data in enumerate(train_loader):  # inner loop within one epoch

            iter_start_time = time.time()  # timer for computation per iteration
            t_data = iter_start_time - iter_data_time
            total_iters += 1
            epoch_iter += 1
            model.set_input(data, epoch=epoch, net_1=w_1_oracle, net_x=w_x, net_2=w_2,
                            net_1_o=w_1_oracle)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
            model.eval_stats()

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

        model.update_learning_rate()
        model.reset_eval_stats()
        for i, data in enumerate(eval_loader):  # inner loop within one epoch
            model.set_input(data, epoch=epoch, net_1=w_1_oracle, net_x=w_x, net_2=w_2,
                            net_1_o=w_1_oracle)  # unpack data from dataset and apply preprocessing
            model.forward_eval()
            model.eval_stats()
        model.eval_stats(print_stat=True, eval=True)
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    # pretrain the model
    model_pretrain = create_model(opt)      # create a model given opt.model and other options
    model_pretrain.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    dataset_size = len(train_loader.dataset)
    for epoch in range(opt.pretrain_epoch):
        train(model_pretrain, epoch)

    # plug in orthogonal classifier after pretraining stage
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    dataset_size = len(train_loader.dataset)
    # loading pretrained CycleGAN
    model.load_networks(epoch=opt.pretrain_epoch-1, pre_name=opt.name, pre_obj='orthogonal')
    # controlled style transfer
    for epoch in range(opt.pretrain_epoch, opt.n_epochs + opt.n_epochs_decay + 1):
        train(model, epoch)
