# %%
import os 
import sys
import math
# import fire
import json
from torch.functional import split
from tqdm import tqdm
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing

import numpy as np
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F

from torch.optim import Adam
from torch.autograd import grad as torch_grad 

import torchvision
from torchvision import transforms

from linear_attention_transformer import ImageLinearAttention

from PIL import Image
from pathlib import Path

# %%
try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False
# %%
# 自己写的方法
from diff_augment import DiffAugment
# %%
assert torch.cuda.is_available(), 'you need to have an nvidia gpu with cuda installed'

num_cores = multiprocessing.cpu_count() # 返回系统的cpu数
# %%
# constants
EXTS = ['jpg', 'jpeg', 'png', 'webp']
EPS = 1e-8
# %%
# helper classes
class NanException(Exception):
    pass

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if old is None:
            return new 
        return old * self.beta + (1 - self.beta) * new

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

class Residual(nn.Module):
    "剩余的"
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return self.fn(x) + x

class Flatten(nn.Module):
    "变平"
    def __init__(self, index):
        super().__init__()
        self.index = index
    def forward(self, x):
        return x.flatten(self.index)

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g
# %%
# one layer of self-attention and feedforward, for image

attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(Rezero(ImageLinearAttention(chan, norm_queries=True))),
    Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])

# %%
# helpers

def default(value, d):
    return d if value is None else value

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def cast_list(el):
    '''
    把输入的el变成list

    Args:
        el ([type]): [description]

    Returns:
        [list]: 返回一个list对象
    '''    
    return el if isinstance(el, list) else [el]

def is_empty(t):
    '''
    判断输入张量t，是不是空

    Args:
        t ([type]): 输入张量

    Returns:
        [type]: bool
    '''    
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return t is None

def raise_if_nan(t):
    '''
    判断t是不是nan,如果是nan抛出异常

    Args:
        t ([type]): [tensor]

    Raises:
        NanException: 调用exception
    '''    
    if torch.isnan(t):
        raise NanException

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    
    else:
        loss.backward(**kwargs)

def gradient_penalty(images, outputs, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=outputs, inputs=images, 
                        grad_outputs=list(map(lambda t: torch.ones(t.size()).cuda(),
                        outputs)), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def calc_pl_lengths(styles, images):
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape).cuda() / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles, 
                        grad_outputs=torch.ones(outputs.shape).cuda(),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

def noise(n, latent_dim):
    '''
    从torch.randn()中创建tensor，形状是(n, latent_dim)

    Args:
        n ([type]): [description]
        latent_dim ([type]): [description]

    Returns:
        [tensor]: [torch.randn(n, latent_dim)]
    '''    
    return torch.randn(n, latent_dim).cuda()

def noise_list(n, layers, latent_dim):
    '''
    使用noise，然后返回list:[noise, layers]

    Args:
        n ([type]): [description]
        layers ([type]): [description]
        latent_dim ([type]): [description]

    Returns:
        [list]: [noise(n, latent_dim), layers]
    '''    
    return [(noise(n, latent_dim), layers)]

def mixed_list(n, layers, latent_dim):
    '''
    混合的list

    Args:
        n ([type]): [description]
        layers ([type]): [description]
        latent_dim ([type]): [description]

    Returns:
        [type]: [description]
    '''    
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim) + noise_list(n, layers - tt, latent_dim)

def latent_to_w(style_vectorizer, latent_descr):
    '''
    (style_vectorizer(z), num_layers)

    Args:
        style_vectorizer ([function]): [description]
        latent_descr (iter): [description]

    Returns:
        list: [description]
    '''             
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

def image_noise(n, im_size):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda()

def leaky_relu(p=0.2):
    '''
    leakyrelu p=0.2

    Args:
        p (float, optional): [description]. Defaults to 0.2.

    Returns:
        [type]: [description]
    '''    
    return nn.LeakyReLU(p)

def evaluate_in_chunks(max_batch_size, model, *args):

    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

def set_requires_grad(model, bool):
    '''
    设置model中的参数是否计算梯度，bool

    Args:
        model ([type]): [description]
        bool (bool): true or false
    '''    
    for p in model.parameters():
        p.requires_grad = bool


# %%
