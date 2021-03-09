# %%
import argparse
import os 
import numpy as np
import math
import sys
from numpy.core.fromnumeric import transpose
from numpy.lib.npyio import save
from torch.utils.tensorboard.summary import image
import torchvision 

import torchvision.transforms as transforms 
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable, variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.utils.tensorboard import SummaryWriter

sys.path.append('/home/xchen/ComputerVision/utils')
# sys.path.append('H:/ComputerVision/utils')

# %%
os.makedirs('images/wgan_cifar10', exist_ok=True)

# output to ./runs/
writer = SummaryWriter('./runs/cifar10')
# %%
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args([])
print(opt)

# %%
img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
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
# initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# %%
# from UsePlatform import getSystemName

# if getSystemName() == "Windows":
#     dst = datasets.MNIST(
#         "../data/",
#         train=True,
#         download=False,
#         transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
#     )
# if getSystemName() == "Linux":
#     dst = datasets.MNIST(
#         '/home/xchen/data/',
#         train=True,
#         download=False,
#         transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
#     )
dst = datasets.CIFAR10(
    '../GAN_implementatnion/data/',
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(opt.img_size),transforms.Normalize([0.5], [0.5])]
    )
)
# configure data loader 
dataloader = DataLoader(
    dst,
    batch_size= opt.batch_size,
    shuffle=True,
)
# %%
# optimizers 
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# %%
# 把图片放到tensorboard中
# img, label = next(iter(dataloader))

# # grid = torchvision.utils.make_grid(img)
# # writer.add_image('images', grid, 0)
# writer.add_graph(discriminator, img.cuda())

# %%
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
# import LossHistory
save_name = []

# %%
# Training
batch_done = 0
print('training start!')
for epoch in range(opt.n_epochs):

    for i, (imgs, _) in enumerate(dataloader):
        
        D_losses = []
        G_losses = []

        # configure input 
        real_imgs = Variable(imgs.type(Tensor)) # 这个方法要被废弃了
        # real_imgs = imgs.type(Tensor).requires_grad = True

        # train discriminator 

        optimizer_D.zero_grad()

        # sample noise as generator input 
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        # z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).requires_grad = True

        # generate a batch of images 
        fake_imgs = generator(z).detach()

        # adversarial loss 
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        D_losses.append(loss_D.item())
        writer.add_scalar('epoch/D_loss_with_epoch', loss_D, epoch)
        # writer.add_scalar('iter/D_loss_with_iter', loss_D, i)

        # clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # train generator
            optimizer_G.zero_grad()

            # generator a batch of images 
            gen_imgs = generator(z)

            # adversarial loss 
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()
            
            G_losses.append(loss_G.item())
            writer.add_scalar('epoch/G_loss_with_epoch', loss_G, epoch)
            # writer.add_scalar('iter/G_loss_with_iter', loss_G, i)
        
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batch_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )
        
        grid = torchvision.utils.make_grid(imgs)
        writer.add_image('images',grid, 0)
    

        if batch_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/wgan_cifar10/%d.png" % batch_done, nrow=5, normalize=True)
            save_name.append(batch_done)
        batch_done += 1



    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

    LossHistory.show_train_animation('./images/wgan/', save_name)

# %%
writer.close()
LossHistory.show_train_hist(train_hist, save=True, path='./images/wgan/train_hist.png')





