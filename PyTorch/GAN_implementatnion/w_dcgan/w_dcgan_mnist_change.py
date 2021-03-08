# %% 
import argparse
import os 
import sys
import numpy as np
from torch.nn.modules.batchnorm import BatchNorm2d
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

os.makedirs("images", exist_ok=True)

# 设置gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# %%
# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

opt = parser.parse_args([])
print(opt)
# %%
cuda = True if torch.cuda.is_available() else False
# output to ./runs/
writer = SummaryWriter()
# %%
def weight_init_normal(m):
    '''
    根据不同的layer初始化参数

    Args:
        m (tensor): layer
    '''    
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
# %%
img_shape = (opt.channels, opt.img_size, opt.img_size)
# %%
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main_module = nn.Sequential(
            # z latent vector 100
            nn.ConvTranspose2d(100, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # 1024, 4, 4
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 512, 8, 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 256, 16, 16
            nn.ConvTranspose2d(256, opt.channels, 4, 2, 1)
        )
        # output of main module > image c x 32 x 32

        self.output = nn.Tanh()

    def forward(self, img):
        img = self.main_module(img)
        return self.output(img)
        
# %%
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main_module = nn.Sequential(
            # image c, 32, 32
            nn.Conv2d(opt.channels, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state 256, 16, 16
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state 512, 8, 8
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # output of main module > state (1024, 4, 4)

        self.output = nn.Sequential(
            # do not apply sigmoid
            nn.Conv2d(1024, 1, 4, 1, 0)
        )

    def forward(self, img):
        img = self.main_module(img)
        return self.output(img)

# %%
# initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# initialize weights
generator.apply(weight_init_normal) # 将weight_init_normal 应用到每个layer
discriminator.apply(weight_init_normal)

# %%
# configure data loader 
dataloader = DataLoader(
    datasets.MNIST(
        "../data/",
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size = opt.batch_size,
    shuffle = True,
)
# %%
# optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# %%
# img, _ = next(iter(dataloader))

# writer.add_graph(discriminator, img.cuda())
# %%
# Training
batch_done = 0
print('Training start!')
for epoch in range(opt.n_epochs):
    
    for i, (imgs, _) in enumerate(dataloader):

        # configure input
        real_imgs = Variable(imgs.type(Tensor))
        # real_imgs = imgs.type(Tensor)
        # real_imgs.requires_grid = True

        # train discriminator

        optimizer_D.zero_grad()

        # sample noise as generator input
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim, 1, 1))).cuda() # 64, 100, 1, 1
        z.requires_grad = True

        # generate a batch of images 
        fake_imgs = generator(z).detach()

        # adversarial loss 
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # 记录loss
        writer.add_scalar('epoch/D_loss_with_epoch', loss_D, epoch)
        writer.add_scalar('iter/D_loss_with_iter/' + str(epoch), loss_D, i)

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

            writer.add_scalar('epoch/G_loss_with_epoch', loss_G, epoch)
            writer.add_scalar('iter/G_loss_with_iter/' + str(epoch), loss_G, i)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batch_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

            writer.add_scalars('iter/loss_' + str(epoch), {'G_loss':loss_G, 'D_loss':loss_D}, i)

        writer.add_scalars('epoch/loss', {'G_loss':loss_G, 'D_loss':loss_D}, epoch)
        # save the gen imgs into tensorboard
        grid = torchvision.utils.make_grid(gen_imgs)
        writer.add_image('images', grid, 0)

        # 每400个图片保存一次生成的图片
        if batch_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], 'images/%d.png' % batch_done, nrow=5, normalize=True)
        batch_done += 1

# %%
writer.close()