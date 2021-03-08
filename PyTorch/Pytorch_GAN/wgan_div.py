# %%
import argparse
import os
import numpy as np
import math
import sys
from torch._C import autocast_decrement_nesting
from torch.functional import Tensor
import torchvision

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
# %%
os.makedirs("images/wgan_div", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args([])
print(opt)

# %%
img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
writer = SummaryWriter(log_dir='runs/wgan_div', comment='wgan_with_div')
# %%
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img
# %%
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
# %%
k = 2
p = 6
# %%
# initialize generator and generator 
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# %%
# configure data loader 
dst = datasets.MNIST(
    "../data",
    train=True,
    download=False,
    transform=transforms.Compose(
        [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    ),
)

dataloader = DataLoader(
    dst,
    batch_size=opt.batch_size,
    shuffle=True,
)
# %%
# optimizers 
optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# %%
# training 
print("Start training!")
 
batch_done = 0
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # configure input 
        real_imgs = imgs.type(Tensor)
        real_imgs.requires_grad = True

        # train discriminator 
        optimizer_D.zero_grad()

        # sample noise as generator input 
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))
        z.requires_grad = True

        # generate a batch of images 
        fake_imgs = generator(z)

        # real images 
        real_validity = discriminator(real_imgs)
        # fake images 
        fake_validity = discriminator(fake_imgs)

        # compute w-div gradient penalty
        real_grad_out = Tensor(real_imgs.size(0), 1).fill_(1.0)

        real_grad = autograd.grad(
            real_validity, real_imgs, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

        fake_grad_out = Tensor(fake_imgs.size(0), 1).fill_(1.0)

        fake_grad = autograd.grad(
            fake_validity, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

        # adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp

        d_loss.backward()
        optimizer_D.step()

        writer.add_scalar('epoch', d_loss, epoch)

        optimizer_G.zero_grad()

        # train the generator every n_critic steps 
        if i % opt.n_critic == 0:

            # train generator 

            # generator a batch of images 
            fake_imgs = generator(z)
            # loss measures generator's ability to fool the discriminator 
            # train on fake images 
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            writer.add_scalar('epoch', g_loss, epoch)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            if batch_done % opt.sample_interval == 0:
                save_image(fake_imgs.data[:25], "images/%d.png" % batch_done, nrow=5, normalize=True)

            grid = torchvision.utils.make_grid(fake_imgs)
            writer.add_image('fake image', grid, 0)
            batch_done += opt.n_critic