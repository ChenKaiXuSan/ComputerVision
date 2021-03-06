# %%
import os 
import sys
import math
# import fire
import json
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
