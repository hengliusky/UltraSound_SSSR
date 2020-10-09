import itertools
import argparse
import glob

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from pytorch_msssim import SSIM
from PIL import Image
from configs import Config
from us_model import USGan_g, USGan_g2
from yoon_model import High2Low, Discriminator
from vgg import Vgg19
from datasets import TrainDatasets, TestDatasets
from us_utils1 import *

from tensorboardX import SummaryWriter

writer = SummaryWriter('runs')

current_path = os.path.dirname(__file__)

# paramters
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
parser.add_argument('--batchsize', type=int, default=1, help='size of the batches')
parser.add_argument('--crop_size', type=int, default=64, help='size of the data crop (squared assumed)')
parser.add_argument('--decay_epoch', type=int, default=2,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', default=True, action='store_true', help='use GPU computation')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--dataroot', type=str, default=current_path, help='root directory of the dataset')
parser.add_argument('--scale_factor', default=[[4.0, 4.0]], help='starting epoch')
parser.add_argument('--upscale_method', default='cubic', help='starting epoch')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')

# Data augmentation related params
parser.add_argument('--augment_no_interpolate_probability', type=int, default=0.05, help='no_interpolate_probability')
parser.add_argument('--augment_leave_as_is_probability', type=int, default=0.45, help='leave_as_is_probability')
parser.add_argument('--augment_min_scale', type=int, default=0.5, help='min_scale')
parser.add_argument('--augment_scale_diff_sigma', type=int, default=0.25, help='scale_diff_sigma')
parser.add_argument('--augment_shear_sigma', type=int, default=0.1, help='sigma')
parser.add_argument('--augment_allow_rotation', default=True, help='recommended false for non-symmetric kernels')

opt = parser.parse_args()

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Networks
netG_L2H = USGan_g(opt.input_nc, opt.output_nc)
netG_H2L = High2Low()
netD_L = Discriminator(64)
netD_H = Discriminator(64)
net_vgg = Vgg19()

if opt.cuda:
    netG_L2H.cuda()
    netG_H2L.cuda()
    netD_L.cuda()
    netD_H.cuda()
    net_vgg.cuda()

netG_L2H.apply(weights_init_normal)
netG_H2L.apply(weights_init_normal)
# netD_L.apply(weights_init_normal)
# netD_H.apply(weights_init_normal)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_vgg = torch.nn.MSELoss()
criterion_l1 = torch.nn.L1Loss()
ssim_module = SSIM(data_range=255, size_average=True, channel=3)

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_L2H.parameters(), netG_H2L.parameters()), lr=opt.lr,
                               betas=(0.5, 0.999))
optimizer_D_L = torch.optim.Adam(netD_L.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_H = torch.optim.Adam(netD_H.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_L = torch.optim.lr_scheduler.LambdaLR(optimizer_D_L,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_H = torch.optim.lr_scheduler.LambdaLR(optimizer_D_H,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# inputs
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_L = Tensor(opt.batchsize, opt.input_nc, opt.crop_size, opt.crop_size)
input_H = Tensor(opt.batchsize, opt.input_nc, opt.crop_size, opt.crop_size)
target_real = Variable(Tensor(np.ones((opt.batchsize, 1))), requires_grad=False)
target_fake = Variable(Tensor(np.zeros((opt.batchsize, 1))), requires_grad=False)

fake_L_buffer = ReplayBuffer()
fake_H_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [transforms.ToTensor()]


class Cycle_ZSSR:
    def __init__(self, input_img, conf=Config()):
        self.opt = opt
        self.cuda = conf.cuda
        # Read input image (can be either a numpy array or a path to an image file)
        self.input = input_img

        self.max_iters = self.opt.n_epochs
        self.model_L2H = USGan_g(opt.input_nc, opt.output_nc)
        self.model_H2L = High2Low()

        self.hr_fathers_sources = [self.input]

    def run(self):
        print('** Start training for sf=4 **')
        if opt.cuda:
            self.model_L2H.cuda()
            self.model_H2L.cuda()

        # Train network
        self.train()

    def train(self):
        # def losses and optimizer
        # Losses
        criterion_GAN = torch.nn.MSELoss()
        criterion_cycle = torch.nn.L1Loss()
        criterion_l1 = torch.nn.L1Loss()

        for epoch in range(self.max_iters):
            # Use augmentation from original input image to create current father.
            # If other scale factors were applied before, their result is also used (hr_fathers_in)
            self.hr_father = random_augment(ims=self.hr_fathers_sources,
                                            base_scales=[1.0] + self.opt.scale_factors,
                                            leave_as_is_probability=self.opt.augment_leave_as_is_probability,
                                            no_interpolate_probability=self.opt.augment_no_interpolate_probability,
                                            min_scale=self.opt.augment_min_scale,
                                            max_scale=([1.0] + self.opt.scale_factors)[
                                                len(self.hr_fathers_sources) - 1],
                                            allow_rotation=self.opt.augment_allow_rotation,
                                            scale_diff_sigma=self.opt.augment_scale_diff_sigma,
                                            shear_sigma=self.opt.augment_shear_sigma,
                                            crop_size=self.opt.crop_size)

            # Get lr-son from hr-father
            self.lr_son = self.father_to_son(self.hr_father)
            interpolated_lr_son = imresize(self.lr_son, self.sf, self.hr_father.shape, self.opt.upscale_method)
            # should convert input and output to torch tensor

            lr_son_input = torch.Tensor(interpolated_lr_son).permute(2, 0, 1).unsqueeze_(0)
            hr_father = torch.Tensor(hr_father).permute(2, 0, 1).unsqueeze_(0)
            lr_son_input = lr_son_input.requires_grad_()

            if self.opt.cuda == True:
                hr_father = hr_father.cuda()
                lr_son_input = lr_son_input.cuda()

            train_output = self.model_L2H(lr_son_input)


if __name__ == "__main__":
    img = np.random(1, 3, 64, 64)
    cz = Cycle_ZSSR(img)
    cz.train()



