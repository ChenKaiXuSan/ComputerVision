# %%
import os 
import sys
import math
# import fire
import json
from torch.functional import split
from torch.nn.modules.conv import Conv2d
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

from linear_attention_transformer import ImageLinearAttention, images

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

def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val ) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res 

def warmup(start, end, max_steps, current_step):
    if current_step > max_steps:
        return end
    return  (end - start) * (current_step / max_steps) + start

def log(t, eps = 1e-6):
    return torch.log(t + eps)

def cutmix_coordinates(height, width, alpha = 1.):
    lam = np.random.beta(alpha, alpha)

    cx = np.random.uniform(0, width)
    cy = np.random.uniform(0, height)
    w = width * np.sqrt(1 - lam)
    h = height * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, width)))    
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, height)))

    return ((y0, y1), (x0, x1)), lam

def cutmix(source, target, coors, alpha = 1.):
    source, target = map(torch.clone, (source, target))
    ((y0, y1), (x0, x1)), _ = coors
    source[:, :, y0:y1, x0:x1] = target[:, :, y0:y1, x0:x1]
    return source

def mask_src_tgt(source, target, mask):
    return source * mask + (1 - mask) * target

# %%
# dataset 

def convert_rgb_to_transparent(image):
    if image.mode == 'RGB':
        return image.convert('RGBA')
    return image

def convert_transparent_to_rgb(image):
    if image.mode == 'RGBA':
        return image.convert('RGB')
    return image 

class expand_greyscale(object):
    def __init__(self, num_channels) -> None:
        super().__init__()
        self.num_chanels = num_channels
    
    def __call__(self, tensor):
        return tensor.expand(self.num_chanels, -1, -1)


def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image 

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, transparent = False, aug_prob = 0.) -> None:
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        num_channels = 3 if not transparent else 4

        self.transform = transforms.Compose({
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), 
                        ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale(num_channels)),
        })
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)
        
# %%
# augmentations

def random_float(lo, hi):
    return lo + (hi - lo) * random()

def random_crop_and_resize(tensor, scale):
    b, c, h, _ = tensor.shape
    new_width = int(h * scale)
    delta = h - new_width
    h_delta = int(random() * delta)
    w_delta = int(random() * delta)
    cropped = tensor[:, :, h_delta:(h_delta + new_width), w_delta:(w_delta + new_width)].clone()
    return F.interpolate(cropped, size=(h, h), mode='bilinear')

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dim=(3, ))

class AugWrapper(nn.Module):
    def __init__(self, D, image_size, types):
        super().__init__()
        self.D = D
        self.types = types 

    def forward(self, images, prob=0., detach = False):
        if random() < prob:
            images = random_hflip(images, prob=0.5)
            images = DiffAugment(images, types= self.types)

        if detach:
            images.detach_()

        return self.D(images), images

# %%
# stylegan2 classes

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias = self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])
    
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod = True, stride = 1, dilation = 1, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel 
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + EPS)
            weights = weights * d
        
        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if prev_rgb is not None:
            x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)

        return x 

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True, upsample_rgb = True, rgba = False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise):
        if self.upsample is not None:
            x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb

def double_conv(chan_in, chan_out):
    '''
    unet结构

    Args:
        chan_in (tensor): [description]
        chan_out (tensor): [description]

    Returns:
        nn.Sequential: conv2d, leakyrelu, conv2d, leakyrelu
    '''    
    return nn.Sequential(
        nn.Conv2d(chan_in, chan_out, 3, padding=1),
        leaky_relu(),
        nn.Conv2d(chan_out, chan_out, 3, padding=1),
        leaky_relu()
    )

class DownBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))

        self.net = double_conv(input_channels, filters)
        self.down = nn.Conv2d(filters, filters, 3, padding=1, stride=2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        unet_res = x

        if self.down is not None:
            x = self.down(x)

        x = x + res
        return x, unet_res

class UpBlock(nn.Module):
    def __init__(self, input_channels, filters):
        super().__init__()
        self.conv_res = nn.ConvTranspose2d(input_channels // 2, filters, 1, stride=2)
        self.net = double_conv(input_channels, filters)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.input_channels = input_channels
        self.filters = filters

    def forward(self, x, res):
        *_, h, w = x.shape
        conv_res = self.conv_res(x, output_size = (h * 2, w * 2))
        x = self.up(x)
        x = torch.cat((x, res), dim=1)
        x = self.net(x)
        x = x + conv_res
        return x 

class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity = 16, transparent=False, no_const = False, fmap_max = 512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * 2 (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs =  zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initaial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initaial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))
        
        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)

        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan)
            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample=not_first,
                upsample_rgb=not_last,
                rgba=transparent,
            )

            self.blocks.append(block)

    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initaial_block(avg_style)
        else:
            x = self.initaial_block.expand(batch_size, -1, -1, -1)
        
        x = self.initial_conv(x)
        styles = styles.transpose(0,1)

        rgb = None 
        for style, block, attn in zip(styles, self.blocks, self.attns):
            if attn is not None:
                x = attn(x)
            
            x, rgb = block(x, rgb, style, input_noise)

        return rgb

# %%
class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity = 16, transparent = False, fmap_max = 512):
        super().__init__()
        num_layers = int(log2(image_size) - 3)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        filters[-1] = filters[-2]

        chan_in_out = list(zip(filters[:-1], filters[1:]))
        chan_in_out = list(map(list, chan_in_out))

        down_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DownBlock(in_chan, out_chan, downsample=is_not_last)
            down_blocks.append(block)

            attn_fn = attn_and_ff(out_chan)
            attn_blocks.append(attn_fn)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)

        last_chan = filters[-1]

        self.to_logit = nn.Sequential(
            leaky_relu(),
            nn.AvgPool2d(image_size // (2 ** num_layers)),
            Flatten(1),
            nn.Linear(last_chan, 1)
        )

        self.conv = double_conv(last_chan, last_chan)

        dec_chan_in_out = chan_in_out[:-1][::-1]
        self.up_blocks = nn.ModuleList(list(map(lambda c: UpBlock(c[1] * 2, c[0]), dec_chan_in_out)))
        self.conv_out = nn.Conv2d(3, 1, 1)

        
        # %%
