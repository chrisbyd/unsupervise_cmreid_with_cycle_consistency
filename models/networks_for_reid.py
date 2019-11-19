import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################
class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer



def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """

    init_weights(net, init_type, init_gain=init_gain)
    return net

def define_H(input_nc, feature_dim ,ngf, norm='batch',init_type='normal',init_gain=0.02):
    norm_layer = get_norm_layer(norm_type=norm)
    net = FeatureExtractionNet(input_nc,feature_dim,ngf,norm_layer,False)
    return init_net(net,init_type,init_gain)


def define_G(feature_dim, output_nc,ngf,norm='batch'):
    norm_layer = get_norm_layer(norm_type=norm)
    net = FakeImageGeneratorNet(output_nc=output_nc,feature_dim=feature_dim,ngf=ngf,norm_layer=norm_layer)
    return init_net(net,init_type='normal',init_gain=0.02)

def define_D(input_nc,ndf,norm='batch', init_type='normal', init_gain=0.02):
    norm_layer = get_norm_layer(norm)
    net = NLayerDiscriminator(input_nc,ndf,n_layers=3,norm_layer=norm_layer)
    return init_net(net,init_type,init_gain)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            # print("THe prediction shape is\n")
            # print(prediction.shape)
            # print("The target tensor shape is\n")
            # print(target_tensor.shape)

            loss = self.loss(prediction, target_tensor)

        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class HingeLoss(nn.Module):

    def __init__(self,margin):
        super(HingeLoss,self).__init__()
        self.margin =margin


    def square_loss(self,feature_A,feature_B):
        diff = feature_A -feature_B
        dis = torch.sum(torch.mul(diff,diff),1)
        return dis

    def forward(self, feature_A,feature_B):
        dis = self.square_loss(feature_A,feature_B)
        hinge_loss = torch.clamp(dis,min=self.margin)
        loss = torch.mean(hinge_loss)
        return loss



class FakeImageGeneratorNet(nn.Module):
    """
    Create a Image generator
    """
    def __init__(self,output_nc,feature_dim ,ngf=64,norm_layer=nn.BatchNorm2d, n_blocks=4):
        assert (n_blocks >=0)
        super(FakeImageGeneratorNet,self).__init__()
        self.pool_dim=512
        model_1 = [nn.Linear(feature_dim,self.pool_dim*9*5)]

        model_2 =[
              norm_layer(self.pool_dim),
              nn.ConvTranspose2d(self.pool_dim,256,stride=2,padding=1,kernel_size=3,output_padding=(1,0)),
              norm_layer(256),
              nn.ReLU(True)
        ]
        model_2 += [
            nn.ConvTranspose2d(256,128,stride=2,padding=1,kernel_size=3,output_padding=1),
            norm_layer(128),
            nn.ReLU(True)
        ]
        model_2 += [
            nn.ConvTranspose2d(128,64,stride=2,padding=1,kernel_size=3,output_padding=1),
            norm_layer(64),
            nn.ReLU(True)
        ]
        model_2 +=[
            nn.Upsample(scale_factor=2)
        ]
        model_2 += [
            nn.ConvTranspose2d(64,ngf,stride=2,padding=1,kernel_size=3,output_padding=1),
            norm_layer(ngf),
            nn.ReLU(True)
        ]
        model_2 +=[
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf,output_nc,kernel_size=7,padding=0),
            nn.Tanh()
        ]
        self.model_1 =nn.Sequential(*model_1)
        self.model_2 = nn.Sequential(*model_2)


    def forward(self, input):
        output_1 = self.model_1(input)

        resize = output_1.view(output_1.shape[0],self.pool_dim,9,5)

        fake_img = self.model_2(resize)

        return fake_img

class  FeatureExtractionNet(nn.Module):
    'Create a feature extraction network'
    #input size [batch_size,3,288,144]
    def __init__(self,input_nc,feature_dim,ngf=64,norm_layer =nn.BatchNorm2d,n_blocks =4):
        assert(n_blocks >=0)
        super(FeatureExtractionNet,self).__init__()

        #output [batch_size,64,144,72]

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc,ngf,kernel_size=7,stride=2,padding=0,bias=True),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        #output [batch_size,64,72,36]
        model += [nn.MaxPool2d(kernel_size=3,stride=2,padding=1)]
        n_downsampling = 2

        #model [batch_size,512,9,5]
        model += [BasicBlock(ngf,ngf,stride=1),
                  BasicBlock(ngf,ngf*2,stride=2),
                  BasicBlock(ngf*2,ngf*4,stride=2),
                  BasicBlock(ngf*4,ngf*8, stride=2)]

        model += [nn.AdaptiveAvgPool2d((1,1))]

        pool_dim = 512
        FC = [nn.Linear(pool_dim,feature_dim)]
        FC += [nn.BatchNorm1d(feature_dim)]

        #model += FC
        self.model = nn.Sequential(*model)
        self.fc= nn.Sequential(*FC)

    def forward(self, input):
        out = self.model(input)
        out = out.view(out.size(0),out.size(1))

        out = self.fc(out)

        return out



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample =None
        if inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes,planes,stride=stride),
                norm_layer(planes)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        #input [batch_size,3,288,144]
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        # [batch_size,144,72]
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # output = [batch_size,1,8,4]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)