from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',help='job launcher')
        self.parser.add_argument('--local_rank', type=int, default=0)

        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=2, help='frequency of saving checkpoints at the end of epochs')        
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training

        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=5, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=5, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--niter_image_train', type=int, default=5, help='# of iter at one image base learning')
        self.parser.add_argument('--is_baseline', action='store_true',  help='# learning of baseline')
        self.parser.add_argument('--progress_train', action='store_true',  help='# scene-wise progressive learning(zoom->walking)')
     
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate for adam')
        self.parser.add_argument('--max_frames_per_gpu', type=int, default=5, help='max number of frames to load into one GPU at a time')
        self.parser.add_argument('--max_t_step', type=int, default=2, help='max spacing between neighboring sampled frames. If greater than 1, the network may randomly skip frames during training.')
        self.parser.add_argument('--n_frames_total', type=int, default=8, help='the overall number of frames in a sequence to train with') 
        self.parser.add_argument('--n_frames_G', type=int, default=3, help='number of input frames to feed into generator, i.e., n_frames_G-1 is the number of frames we look into past')
        self.parser.add_argument('--n_splits_per_video', type=int, default=4, help='number of loading per video. n_loadings_per_video*video num is dataset size')
        self.parser.add_argument('--lambda_t',type=float,default=1, help="temporal hyperparam")
        #self.parser.add_argument('--max_frames_per_gpu', type=int, default=1,help='max number of frames to load into one GPU at a time')


        self.parser.add_argument('--PFAFN_warp_checkpoint', type=str, help='load the pretrained model from the specified location')
        self.parser.add_argument('--PFAFN_gen_checkpoint', type=str,  help='load the pretrained model from the specified location')
        self.parser.add_argument('--PBAFN_warp_checkpoint', type=str, help='load the pretrained model from the specified location')
        self.parser.add_argument('--PBAFN_gen_checkpoint', type=str,  help='load the pretrained model from the specified location')
        

        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        self.isTrain = True


