# %%
import argparse
import os 
import numpy as np
import math
import sys
from torch import autograd
import torchvision

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optims

from torch.utils.tensorboard import SummaryWriter

# %%
import shutil
# 删除文件夹

os.makedirs("images/wgan_gp", exist_ok=True)
shutil.rmtree("images/wgan_gp")
os.makedirs("images/wgan_gp", exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# %%
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

writer = SummaryWriter()
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
            nn.Tanh()
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
# loss weight for gradient penalty 
lambda_gp = 10

# initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
# %%
# configure data loader 
dataloader = DataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)
# %%
# optimizers
optimizer_G = optims.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = optims.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# %%
def compute_gradient_penalty(D, real_samples, fake_samples):
    '''
    calculates the gradient penalty loss for wgan gp

    Args:
        D (model): discriminator
        real_samples (tensor): real samples.data
        fake_samples (tensor): fake samples.data

    Returns:
        tensor: return tensor
    '''
    # random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # get random interpolation between real and fake samples 
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = True

    # get gradient w,r,t interpolates
    gradients = autograd.grad(
        outputs = d_interpolates,
        inputs = interpolates, 
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
# %%
# Training
batches_done = 0
print("Training start!")
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # configure input
        real_imgs = imgs.type(Tensor)
        real_imgs.requires_grad = True

        # Train Discriminator

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
        # gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data) 
        # adversarial loss 
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        writer.add_scalar("epoch", d_loss, epoch)
        writer.add_scalar("iter", d_loss, i)

        optimizer_G.zero_grad()

        # train the generator every n_critic steps 
        if i % opt.n_critic == 0:

            # train generator

            # generate a batch of images 
            fake_imgs = generator(z)
            # loss measures generator's ability to fool the discriminator
            # train on fake images 
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            writer.add_scalar("epoch", g_loss, epoch)
            writer.add_scalar("iter", g_loss, i)
            writer.add_scalars("epoch", {'g_loss':g_loss, 'd_loss':d_loss}, epoch)
            writer.add_scalars("iter", {'g_loss':g_loss, 'd_loss':d_loss}, i)          

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            
            if batches_done % opt.sample_interval == 0:
                save_image(fake_imgs.data[:25], "images/wgan_gp/%d.png" % batches_done, nrow=5, normalize=True)
                grid = torchvision.utils.make_grid(fake_imgs)
                writer.add_image('fake image', grid, 0)

            batches_done += opt.n_critic

# %%
writer.close()


