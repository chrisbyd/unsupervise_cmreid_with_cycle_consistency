import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel

from . import networks_for_reid as networks

class CycleReidModel(BaseModel):
    """
      This class implements the CycleGAN model, for learning image-to-image translation without paired data.
      It is further tailored to fit the person re-identification purposes
      """
    def __init__(self,config):
        """
        Initialize the Cycle-reid model
        :param config:
        """
        BaseModel.__init__(self,config)
        self.config = config
        self.loss_names = ['D_A','G_A','cycle_A','idt_A','D_B','G_B','cycle_B','idt_B']
        #specify the images to show
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names =visual_names_A +visual_names_B
        if self.isTrain:
            self.model_names = ['G_A', 'H_A','G_B','H_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'H_A','G_B','H_B']

        # define networks (both Generators and discriminators)
        # The basic logic flow is like this Real_a -> H_A -> G_A -> Fake_B
        # Real_B -> H_B -> G_B -> Fake_A
        self.netH_A = networks.define_H(config.input_nc, config.feature_dim ,config.ngf).to(self.device)

        self.netG_A = networks.define_G(config.feature_dim,config.output_nc,config.ngf).to(self.device)

        self.netH_B = networks.define_H(config.output_nc, config.feature_dim,config.ngf).to(self.device)

        self.netG_B = networks.define_G(config.feature_dim,config.input_nc,config.ngf).to(self.device)
        if self.isTrain:
            self.netD_A = networks.define_D(config.output_nc,config.ngf).to(self.device)

            self.netD_B =networks.define_D(config.input_nc,config.ngf).to(self.device)

        if self.isTrain:
            if config.lambda_identity > 0.0:
                assert (config.input_nc == config.output_nc)

            self.fake_A_pool = ImagePool(config.pool_size)
            self.fake_B_pool = ImagePool(config.pool_size)

            #define the loss function
            self.criterionGAN = networks.GANLoss(config.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            self.hinge_loss =networks.HingeLoss(margin= config.margin).to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(),
                                                                self.netG_B.parameters(),
                                                                self.netH_A.parameters(),
                                                                self.netH_B.parameters()),
                                                lr=config.lr, betas=(config.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=config.lr, betas=(config.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        self.global_num =1

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

                Parameters:
                    input (dict): include the data itself and its metadata information.

                The option 'direction' can be used to swap domain A and domain B.
                """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # ---- debug

    def set_test_input(self,input,type):
        if type == 'visible':
            self.real_A = input.to(self.device)
        elif type == 'thermal':
            self.real_B = input.to(self.device)
        else:
            raise NotImplementedError("The feature extractor for this type of images has not been developed")

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real_feature_A = self.netH_A(self.real_A)
        self.fake_B = self.netG_A(self.real_feature_A)  # G_A(A)
        self.fake_feature_B = self.netH_B(self.fake_B)
        self.rec_A = self.netG_B(self.fake_feature_B)   # G_B(G_A(A))
        self.real_feature_B = self.netH_B(self.real_B)
        self.fake_A = self.netG_B(self.real_feature_B)  # G_B(B)
        self.fake_feature_A = self.netH_A(self.fake_A)
        self.rec_B = self.netG_A(self.fake_feature_A)   # G_A(G_B(B))

    #visible net
    def get_feature_A(self):
        real_feature_A = self.netH_A(self.real_A)
        return real_feature_A

    def get_feature_B(self):
        real_feature_B = self.netH_B(self.real_B)
        return real_feature_B

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
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_H(self):

        real_feature_A = self.netH_A(self.real_A)
        fake_feature_B = self.netH_B(self.fake_B.detach())
        real_feature_B = self.netH_B(self.real_B)
        fake_feature_A = self.netH_A(self.fake_A.detach())
        hinge_loss_1 = self.hinge_loss(real_feature_A,fake_feature_B)
        hinge_loss_2 = self.hinge_loss(real_feature_B,fake_feature_A)
        loss_H = hinge_loss_1 + hinge_loss_2

        loss_H.backward()

        self.loss_H = loss_H



    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_feature_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_feature_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's,H_A,H_B gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.backward_H()
        self.optimizer_G.step()  # update G_A and G_B's weights


        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights



        




