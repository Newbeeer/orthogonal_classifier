from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=500, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=12, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=0, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='vanilla', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--dataset', type=str, default='mnist')
        parser.add_argument('--bias', type=float, nargs='+', default=[0.9, 0.8, 0.8])
        parser.add_argument('--orthogonal', action='store_true', )
        parser.add_argument('--holdout_fraction', type=float, default=0.2)
        parser.add_argument('--trial_seed', type=int, default=0,
                            help='Trial number (used for seeding split_dataset and '
                                 'random_hparams).')
        parser.add_argument('--data_dir', type=str, default='../domainbed')
        parser.add_argument('--obj', type=str, default='vanilla',
                            help='the type of generator objective. [vanilla| kl | wgangp]. ')
        parser.add_argument('--image_size', type=int, default=64)
        parser.add_argument('--pretrain', action='store_true', )
        parser.add_argument('--gender', action='store_true', )
        parser.add_argument('--oracle', action='store_true', )
        parser.add_argument('--alg', type=str, default="ERM", help='[ERM | MLDG | Fish | TRM].')
        # Eval config
        parser.add_argument('--real', action='store_true', )
        parser.add_argument('--save', action='store_true', )
        parser.add_argument('--out_path', type=str, default='celeba')
        parser.add_argument('--resume_epoch', type=int, default=6)
        parser.add_argument('--domain', type=int, default=0, help='The domains to be transferred [0| 1]')
        self.isTrain = True
        return parser
