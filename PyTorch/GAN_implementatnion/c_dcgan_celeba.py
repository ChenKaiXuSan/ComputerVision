# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # c-DCGAN with CelebA
# https://github.com/znxlwm/pytorch-MNIST-CelebA-cGAN-cDCGAN/blob/master/pytorch_CelebA_cDCGAN.py

# %%
import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


# %%
# G(z)
class generator(nn.Module):
    # initial
    def __init__(self, d=128):
        super(generator, self).__init__()
        # z, 100
        self.deconv1_1 = nn.ConvTranspose2d(100, d*4, 4, 1, 0)
        self.devonv1_1_bn = nn.BatchNorm2d(d*4)
        # y (label 变为2，female and male)
        self.devonv1_2 = nn.ConvTranspose2d(2, d*4, 4, 1, 0)
        self.devonv1_2_bn = nn.BatchNorm2d(d*4)

        self.devonv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.devonv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.devonv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init 
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.deconv1_1_bn(self.deconv1_1(input)), 0.2)
        y = F.leaky_relu(self.deconv1_2_bn(self.devonv1_2(label)), 0.2)
        
        x = torch.cat([x, y], 1)

        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv(x)), 0.2)
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
        x = F.tanh(self.deconv5(x))

        return x


# %%
class discriminator(nn.Module):
    # initial
    def __init__(self, d = 128):
        super(discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(3, d/2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(2, d/2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)

        x = torch.cat([x, y], 1)
        
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x


# %%
def normal_init(m, mean, std):
    # isinstance() 函数来判断一个对象是否是一个已知的类型
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# %%
# label preprocess
img_size = 64
onehot = torch.zeros(2, 2) # 2 * 2
onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(2, 1), 1).view(2, 2, 1, 1) # 2 2 1 1维度

fill = torch.zeros([2, 2, img_size, img_size])
for i in range(2):
    fill[i, i, :, :] = 1

# print(fill)

with open(r'../data/resized_celebA/gender_label.pkl', 'rb') as fp:
    y_gender_ = pickle.load(fp)

y_gender_ = torch.LongTensor(y_gender_).squeeze() # 在给定的维度上进行压缩


# %%
# fixed noise & label
temp_z0_ = torch.randn(4, 100)
temp_z0_ = torch.cat([temp_z0_, temp_z0_], 0)
temp_z1_ = torch.randn(4, 100)
temp_z1_ = torch.cat([temp_z1_, temp_z1_], 0)

fixed_z_ = torch.cat([temp_z0_, temp_z1_], 0)
fixed_y_ = torch.cat([torch.zeros(4), torch.ones(4), torch.zeros(4), torch.ones(4)], 0).type(torch.LongTensor).squeeze()

fixed_z_ = fixed_z_.view(-1, 100, 1, 1)
fixed_y_label = onehot[fixed_y_]
fixed_z_, fixed_y_label = Variable(fixed_z_.cuda(), volatile = True), Variable(fixed_y_label.cuda(), volatile=True)


# %%
def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    G.eval()
    test_images = G(fixed_z_, fixed_y_label)
    G.train()

    size_figure_grid = 4
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla() # clear axis
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transposes(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center') # add text to the axes
    
    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# %%
def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# %%
def show_noise_morp(show=False, save=False, path='result.png'):
    source_z_ = torch.rand(10, 100)
    z_ = torch.zeros(100, 100)
    for i in range(5):
        for j in range(5):
            z_[i*20 + j] = (source_z_[i*2+1] - source_z_[i*2]) / 9 *(j+1) + source_z_[i*2]

    for i in range(5):
        z_[i*20 + 10:i*20 + 20] = z_[i*20 : i*20+10]

    y_ = torch.cat([torch.zeros(10, 1), torch.ones(10, 1)], 0).type(torch.LongTensor).squeeze()
    y_ = torch.cat([y_, y_, y_, y_, y_], 0)
    y_label_ = onehot[y_]
    z_ = z_.view(-1, 100, 1, 1)
    y_label_ = y_label_.view(-1, 2, 1, 1)

    z_, y_label_ = Variable(z_.cuda(), volatile=True), Variable(y_label_.cuda(), volatile=True)

    G.eval()
    test_images = G(z_, y_label_)
    G.train()

    size_figure_grid = 10
    fig, ax = plt.subplots()


# %%
x = torch.rand(3)
print(x)


