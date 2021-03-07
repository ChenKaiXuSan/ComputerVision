# %%
# https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_clipping.py
# %%
import torch
import torch.nn as nn 
import time as t 
import matplotlib.pyplot as plt
import torch.optim as optim
plt.switch_backend('agg')

import os 
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter, writer
# %%
class Generator(nn.Module):
    def __init__(self, channels):
        super(Generator, self).__init__()

        self.main_module = nn.Sequential(
            # z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # state 1024 x 4 x 4
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # state 512 x 8 x 8
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # state 256 x 16 x 16
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1)
        )
        # output of main module > image (cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

# %%
class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()

        self.main_module = nn.Sequential(
            # image (c x 32 x 32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            # image 256 x 16 x 16
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),

            # image 512 x 8 x 8
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # output of main module > state (1024 x 4 x 4)

        self.output = nn.Sequential(
            # do not apply sigmoid at output fo D
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, padding=0)           
        )

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


# %%
class WGAN_CP(object):
    def __init__(self, args):
        super().__init__()
        print("wgan_cp init model.")
        self.G = Generator(args.channels)
        self.D = Discriminator(args.channels)
        self.C = args.channels

        # check cuda 
        self.check_cuda(args.cuda)

        # wgan lr
        self.lr = 0.00005

        self.batch_size = 64
        self.weight_cliping_limit = 0.01

        # optimizer with rmsprop
        self.d_optimizer = optim.RMSprop(self.D.parameters(), lr=self.lr)
        self.g_optimizer = optim.RMSprop(self.G.parameters(), lr=self.lr)

        # set tensorboard
        writer = SummaryWriter()

        self.generator_iters = args.generator_iters
        self.critic_iter = 5

    def check_cuda(self, cuda_flag=False):
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda()
            self.G.cuda()
            print("Cuda enabled flag: {}".format(self.cuda))