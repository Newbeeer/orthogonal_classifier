import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import imageio
from torch_ema import ExponentialMovingAverage
from torchvision.models.inception import inception_v3
from utils import save_image
cross_entropy_loss = nn.CrossEntropyLoss()

def orthogonal_loss(x, y, pred):

    pred = torch.sigmoid(pred)
    w = y * x + (1-y) * 1/x
    w = w.unsqueeze(1)
    density_ratio = (pred*w)/(pred*w + 1 - pred)
    loss = - torch.log(density_ratio + 1e-5)

    return torch.sum(loss) / len(pred)

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')


        self.opt = opt
        self.p_partial_B = self.p_partial_A = None
        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        opt.dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        opt.dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.ema_G_A = ExponentialMovingAverage(self.netG_A.parameters(), decay=0.995)
        self.ema_G_B = ExponentialMovingAverage(self.netG_B.parameters(), decay=0.995)
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.ema_D_A = ExponentialMovingAverage(self.netD_A.parameters(), decay=0.995)
            self.ema_D_B = ExponentialMovingAverage(self.netD_B.parameters(), decay=0.995)
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        self.cnt = 0
        self.cnt_incept = 0
        self.pred = np.zeros((200000, 1000))
        self.save_cnt = 0

    def set_input(self, input, **kwargs):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        img = input[0]
        label = input[1]
        self.real_A = img[label == 0].cuda()
        self.real_B = img[label == 1].cuda()
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

        self.w_2 = kwargs['net_2']
        self.w_1 = kwargs['net_1']
        self.w_x = kwargs['net_x']
        self.w_1_oracle = kwargs['net_1_o']
        self.epoch = kwargs['epoch']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        # print("sanity check the generator:", self.real_A.size(), self.fake_B.size())
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def forward_eval(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.netG_A.eval()
        self.netG_B.eval()
        with torch.no_grad():
            self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)

        self.netG_A.train()
        self.netG_B.train()

    def generate(self, save=False, total_save=5):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        img_dict = {
            'real_A': self.real_A,
            'real_B': self.real_B,
            'fake_A': self.fake_A,
            'fake_B': self.fake_B
        }

        self.cnt_img = {key: 0 for key in img_dict.keys()}
        for key, item in img_dict.items():
            img = item
            for idx in range(len(img)):
                image = img[idx].permute(1, 2, 0).squeeze().cpu().numpy()
                image = ((image + 1.) * 127.5).astype(np.uint8)
                imageio.imwrite(os.path.join('style_transfer/generated_images',
                                             self.opt.out_path, str(self.cnt) + f'_{key}.jpg'), image)
                self.cnt_img[key] += 1

        # save images for visualization
        if save:
            from functools import partial
            fn = partial(os.path.join, './vis_img')
            N = 10
            img = self.fake_B[:N]
            img = 0.5 * img + 0.5
            print("image size:", img.size(), "saving to ", fn(self.opt.out_path + '_A.jpg'))
            save_image(img, fn(self.opt.out_path + '_A.jpg'), nrow=len(img))

            N = 10
            img = self.real_A[:N]
            img = 0.5 * img + 0.5
            print("image size:", img.size(), "saving to ",fn(self.opt.out_path + '_real_A' + '.jpg'))
            save_image(img, fn(self.opt.out_path + '_real_A' + '.jpg'), nrow=len(img))

            N = 10
            img = self.fake_A[:N]
            img = 0.5 * img + 0.5
            print("image size:", img.size(), "saving to ", fn(self.opt.out_path + '_B.jpg'))
            save_image(img, fn(self.opt.out_path + '_B.jpg'), nrow=len(img))

            N = 10
            img = self.real_B[:N]
            img = 0.5 * img + 0.5
            print("image size:", img.size(), "saving to ", fn(self.opt.out_path + '_real_B' + '.jpg'))
            save_image(img, fn(self.opt.out_path + '_real_B' + '.jpg'), nrow=len(img))

            self.save_cnt += 1
            if self.save_cnt == total_save:
                print(f"Generate {total_save} images for visualization")
                print(f"Now existing")
                exit(0)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D


    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        if self.opt.obj == 'vanilla' or (self.opt.obj == 'orthogonal' and self.epoch <= self.opt.pretrain_epoch):
            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        else:
            # fixed classifier:
            pred_A = self.netD_A(self.fake_B)
            pred_B = self.netD_B(self.fake_A)
            G_A_label = torch.ones((len(self.fake_B))).long().cuda()
            G_B_label = torch.zeros((len(self.fake_A))).long().cuda()

            def combine_predict(x):
                p_inv = F.softmax(self.w_1.predict(x), dim=1)[:, 0]
                p_full = F.softmax(self.w_x.predict(x), dim=1)[:, 0]
                r = (p_full * (1 - p_inv)) / ((1 - p_full) * p_inv)
                r = torch.clamp(r, 0.1, 9)
                return r

            self.loss_G_A = orthogonal_loss(combine_predict(self.fake_B), G_A_label, pred_A)
            self.loss_G_B = orthogonal_loss(combine_predict(self.fake_A), G_B_label, pred_B)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B \
                      + self.loss_idt_A + self.loss_idt_B

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B

        # prevent exploding gradient
        torch.nn.utils.clip_grad_norm_(self.netG_A.parameters(), 1e-5)
        torch.nn.utils.clip_grad_norm_(self.netG_B.parameters(), 1e-5)
        self.optimizer_G.step()       # update G_A and G_B's weights

        # # EMA
        # self.ema_G_A.update(self.netG_A.parameters())
        # self.ema_G_B.update(self.netG_B.parameters())

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate gradients for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

        # # EMA
        # self.ema_D_A.update(self.netD_A.parameters())
        # self.ema_D_B.update(self.netD_B.parameters())

    def eval_stats(self, print_stat=False, eval=False):

        # update the statistics of spurious features on the four domains
        # this function is called when opt.eval == True
        # ideally, p_real_A and p_fake_B should have the same statistics
        if eval:
            print("Evaluation....")
        if print_stat:
            print('A: Z_2 acc:{0:.3f}, Z_1 acc:{1:.3f} / B: Z_2 acc:{2:.3f}, Z_1 acc:{3:.3f} /total: Z_2 acc:{4:.3f}, Z_1 acc:{5:.3f}'.format(
                                                                          self.cnt_dict['correct_A'] / self.cnt_dict['rAc'],
                                                                          self.cnt_dict['correct_inv_A'] / self.cnt_dict['rAc'],
                                                                          self.cnt_dict['correct_B'] / self.cnt_dict['rBc'],
                                                                          self.cnt_dict['correct_inv_B'] / self.cnt_dict['rBc'],
                                                                          self.cnt_dict['correct'] / self.cnt_dict['cnt'],
                                                                          self.cnt_dict['correct_inv'] / self.cnt_dict['cnt'],))
            print("Total:", self.cnt_dict['cnt'])
        else:
            real_B = self.real_B
            real_A = self.real_A
            fake_B = self.fake_B
            fake_A = self.fake_A

            correct_A = (self.w_2.predict(real_A).argmax(1) == self.w_2.predict(fake_B).argmax(
                1)).float().sum()
            correct_B = (self.w_2.predict(real_B).argmax(1) == self.w_2.predict(fake_A).argmax(
                1)).float().sum()
            correct = correct_A + correct_B

            correct_inv_A = (self.w_1_oracle.predict(real_A).argmax(1) != self.w_1_oracle.predict(fake_B).argmax(
                1)).float().sum()
            correct_inv_B = (self.w_1_oracle.predict(real_B).argmax(1) != self.w_1_oracle.predict(fake_A).argmax(
                1)).float().sum()
            correct_inv = correct_inv_A + correct_inv_B


            self.cnt_dict['rAc'] += len(real_A)
            self.cnt_dict['rBc'] += len(real_B)
            self.cnt_dict['fAc'] += len(fake_B)
            self.cnt_dict['fBc'] += len(fake_A)
            self.cnt_dict['correct'] += correct
            self.cnt_dict['correct_A'] += correct_A
            self.cnt_dict['correct_B'] += correct_B
            self.cnt_dict['correct_inv'] += correct_inv
            self.cnt_dict['correct_inv_A'] += correct_inv_A
            self.cnt_dict['correct_inv_B'] += correct_inv_B
            self.cnt_dict['cnt'] += len(real_A) + len(real_B)

    def reset_eval_stats(self):
        self.cnt_dict = {'rAc': 0., 'rBc': 0, 'fAc': 0., 'fBc': 0.,
                         'rA': 0., 'rB': 0., 'fA': 0., 'fB': 0., 'correct': 0.,
                         'cnt': 0., 'correct_inv': 0., 'correct_A': 0.,
                         'correct_B':0., 'correct_inv_A': 0., 'correct_inv_B': 0.}