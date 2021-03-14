# %% 
import argparse
import os 
import sys
from PIL.ImageFont import truetype
import numpy as np
from torch import tensor
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.serialization import save
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

os.makedirs("../images/w_dcgan_mnist", exist_ok=True)

# 设置gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# %%
# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
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
writer = SummaryWriter('runs/w_dcgan_mnist')
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
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 512, 8, 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 256, 16, 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 128, 32, 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64, 64, 64
            nn.ConvTranspose2d(64, opt.channels, 4, 2, 1)
        )
        # output of main module > image c x 128 x 128

        self.output = nn.Tanh()

    def forward(self, img):
        img = self.main_module(img)
        return self.output(img)
        
# %%
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main_module = nn.Sequential(
            # image c, 128, 128
            nn.Conv2d(opt.channels, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64, 64, 64
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 128, 32, 32
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state 256, 16, 16
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state 512, 8, 8
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # output of main module > state (1024, 4, 4)

        self.output = nn.Sequential(
            # do not apply sigmoid
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False)
        )

    def forward(self, img):
        img = self.main_module(img)
        img = self.output(img)
        img = img.mean(0)
        return img.view(1)

# %%
import random
manualSeed = random.randint(1, 10000)
print("random seed: ", manualSeed)
torch.manual_seed(manualSeed)

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# %%
# initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

input = torch.FloatTensor(opt.batch_size, opt.latent_dim, opt.img_size, opt.img_size)
noise = torch.FloatTensor(opt.batch_size, opt.latent_dim, 1, 1)
fixed_noise = torch.FloatTensor(opt.batch_size, opt.latent_dim, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1

if cuda:
    generator.cuda()
    discriminator.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

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
print(generator)
print(discriminator)

print('Training start!')
for epoch in range(opt.n_epochs):
    
    for i, (imgs, _) in enumerate(dataloader):

        # configure input
        if cuda:
            real_imgs = imgs.cuda()
        
        input.resize_as_(real_imgs).copy_(real_imgs)
        inputv = input.clone().detach().requires_grad_(True)

        # reset requires_grad
        for p in discriminator.parameters():
            p.requires_grad = True # they are set to false below in netG update


        # train discriminator

        optimizer_D.zero_grad()

        # train with real
        errD_real = discriminator(inputv)
        errD_real.backward(one)

        # train with fake
        noise.resize_(opt.batch_size, opt.latent_dim, 1, 1).normal_(0, 1)
        noisev = noise.clone().detach().requires_grad_(True)

        # generate a batch of images 
        fake = generator(noisev).data
        fake.requires_grad = True

        errD_fake = discriminator(fake)
        errD_fake.backward(mone)

        # adversarial loss 
        errD = errD_real - errD_fake
        optimizer_D.step()

        # 记录loss
        writer.add_scalar('epoch/D_loss_with_epoch', errD, epoch)
        # writer.add_scalar('iter/D_loss_with_iter/' + str(epoch), loss_D, i)

        # clip weights of discriminator 
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            for p in discriminator.parameters():
                p.requires_grad = False # to avoid computation

            # train generator
            optimizer_G.zero_grad()

            # make sure we feed a full batch of noise
            noise.resize_(opt.batch_size, opt.latent_dim, 1, 1).normal_(0, 1)
            noisev = noise.clone().detach().requires_grad_(True)

            # generator a batch of images 
            fake = generator(noisev)

            # adversarial loss 
            errG = discriminator(fake)
            errG.backward(one)
            
            optimizer_G.step()
            
            writer.add_scalar('epoch/G_loss_with_epoch', errG, epoch)
            # writer.add_scalar('iter/G_loss_with_iter/' + str(epoch), loss_G, i)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batch_done % len(dataloader), len(dataloader), errD.item(), errG.item())
            )

            # writer.add_scalars('iter/loss_' + str(epoch), {'G_loss':errG, 'D_loss':errD}, i)

        writer.add_scalars('epoch/loss', {'G_loss':errG, 'D_loss':errD}, epoch)
        # save the gen imgs into tensorboard
        grid = torchvision.utils.make_grid(fake)
        writer.add_image('images', grid, 0)

        # 每400个图片保存一次生成的图片
        if batch_done % opt.sample_interval == 0:
            save_image(input.data[:25], '../images/w_dcgan_mnist/real_img.png', nrow=5, normalize=True)
            # 把固定的noise放到generator里面生成fake image
            fake = generator(fixed_noise)
            save_image(fake.data[:25], '../images/w_dcgan_mnist/%d.png' % batch_done, nrow=5, normalize=True)
        batch_done += 1

# %%
writer.close()