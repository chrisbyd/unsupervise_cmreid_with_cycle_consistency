import argparse
import os

class Config():
    """
    This class defines the configuration during the training and test
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
        self.parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        self.parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
        self.parser.add_argument('--arch', default='resnet50', type=str,
                            help='network baseline:resnet18 or resnet50')
        self.parser.add_argument('--resume', '-r', default='', type=str,
                            help='resume from checkpoint')
        self.parser.add_argument('--test-only', action='store_true', help='test only')
        self.parser.add_argument('--model_path', default='save_model/', type=str,
                            help='model save path')
        self.parser.add_argument('--save_epoch', default=20, type=int,
                            metavar='s', help='save model every 10 epochs')
        self.parser.add_argument('--log_path', default='log/', type=str,
                            help='log save path')
        self.parser.add_argument('--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        self.parser.add_argument('--low-dim', default=512, type=int,
                            metavar='D', help='feature dimension')
        self.parser.add_argument('--img_w', default=144, type=int,
                            metavar='imgw', help='img width')
        self.parser.add_argument('--img_h', default=288, type=int,
                            metavar='imgh', help='img height')
        self.parser.add_argument('--batch-size', default=32, type=int,
                            metavar='B', help='training batch size')
        self.parser.add_argument('--test-batch', default=64, type=int,
                            metavar='tb', help='testing batch size')
        self.parser.add_argument('--method', default='id', type=str,
                            metavar='m', help='method type')
        self.parser.add_argument('--drop', default=0.0, type=float,
                            metavar='drop', help='dropout ratio')
        self.parser.add_argument('--trial', default=1, type=int,
                            metavar='t', help='trial (only for RegDB dataset)')
        self.parser.add_argument('--gpu', default='0', type=str,
                            help='gpu device ids for CUDA_VISIBLE_DEVICES')
        self.parser.add_argument('--mode', default='all', type=str, help='all or indoor')

        self.parser.add_argument('--lamda1', default=0.5, type=float,
                            metavar='t', help='the weight for the modal-specific loss')
        self.parser.add_argument('--lamda2', default=0.6, type=float, help='the weight for the identity loss')

        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')

        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')

        self.parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other 0.1')

        self.parser.add_argument('--input_nc', type=int, default=3,
                            help='# of input image channels: 3 for RGB and 1 for grayscale')
        self.parser.add_argument('--output_nc', type=int, default=3,
                            help='# of output image channels: 3 for RGB and 1 for grayscale')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')

        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')

        self.parser.add_argument('--netD', type=str, default='basic',
                            help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')

        self.parser.add_argument('--netG', type=str, default='resnet_9blocks',
                            help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')

        self.parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')

        self.parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')

        self.parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # dataset parameters
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned',
                            help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')

        self.parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')

    def parse(self):
        return  self.parser.parse_args()

