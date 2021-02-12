# %% 
import torch
from torch._C import CharStorageBase
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy

# %%
class EncoderBlock(nn.Module):
    '''
    encoder block (used in encoder and discriminator)

    Args:
        nn ([type]): [description]
    '''
    def __init__(self, channel_in, channel_out):
        super(self, EncoderBlock).__init__() 

        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, 
                            kernel_size=5, padding=2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, ten, out=False, t=False):
        '''
        here we want to be able to take an intermediate output for reconstruction error

        Args:
            ten ([type]): [description]
            out (bool, optional): 判断是否有中间输出. Defaults to False.
            t (bool, optional): [description]. Defaults to False.

        Returns:
            [tensor]]: 返回通过encoderblock的张量
        '''        
        if out:
            ten = self.conv(ten)
            ten_out = ten
            ten = self.bn(ten)
            ten = F.relu(ten, False)
            # ten = nn.ReLU(ten, False)
            return ten, ten_out
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = F.relu(ten, True)
            return ten 


class DecoderBlock(nn.Module):
    '''
    decoder block (used in the decoder)

    Args:
        nn ([type]): [description]
    '''    
    def __init__(self, channel_in, channel_out):
        super(self, DecoderBlock).__init__()

        # transpose convolution to double the dimensions
        self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2, 
                                    stride=2, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten, True)
        # ten = nn.ReLU(ten, True)
        return ten


class Encoder(nn.Module):
    '''
    encoder , made with encoderblock

    Args:
        nn ([type]): [description]
    '''    
    def __init__(self, channel_in=3, z_size=128):
        super(self, Encoder).__init__()
        self.size = channel_in
        layers_list = []
        # the first time 3 -> 64, for every other double the channel size        
        for i in range(3):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=64))
                self.size = 64
            else:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size * 2))
                self.size *= 2

        # final shape Bx256x8x8
        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(
            nn.Linear(in_features= 8 * 8 * self.size, out_features=1024, bias=False),
            nn.BatchNorm1d(num_features=1024, momentum=0.9),
            nn.ReLU(True)
        )
        # two linear to get the mu vector and the diagonal of the log_variance 
        self.l_mu = nn.Linear(in_features=1024, out_features=z_size)
        self.l_var = nn.Linear(in_features=1024, out_features=z_size)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)
        return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)

class Decoder(nn.Module):
    '''
    decoder, made with decoderblock

    Args:
        nn ([type]): [description]
    '''    
    def __init__(self, z_size, size):
        super(self, Decoder).__init__()
        # start from B*z_size
        self.fc = nn.Sequential(
            nn.Linear(in_features=z_size, out_features=8 * 8 * size, bias=False),
            nn.BatchNorm1d(num_features=8 * 8 * size, momentum=0.9),
            nn.ReLU(True)
        )
        self.size = size 
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size))
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size // 2))
        self.size = self.size // 2
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size // 4))
        self.size = self.size // 4

        # final conv to get 3 channels and tanh layer 
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        ))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):

        ten = self.fc(ten)
        ten = ten.view(len(ten), -1, 8, 8)
        ten = self.conv(ten)
        return ten 

    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)

class Discriminator(nn.Module):
    def __init__(self, channel_in=3, recon_level=3):
        super(self, Discriminator).__init__()
        self.size = channel_in
        self.recon_levl = recon_level
        # module list because we need to extract an intermediate output
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        ))
        self.size = 32
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=128))
        self.size = 128
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))
        self.size = 256
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))

        # final fc to get the score (real or fake)
        self.fc = nn.Sequential(
            nn.Linear(in_features=8 * 8 * self.size, out_features=512, bias=False),
            nn.BatchNorm1d(num_features=512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1),
        )

    def forward(self, ten, ten_original, ten_sampled):

        ten = torch.cat((ten, ten_original, ten_sampled), 0)

        for i, lay in enumerate(self.conv): #  self.conv is modulelist
            # take the 9th layer as one of the outputs
            if i == self.recon_levl: # 3
                ten, layer_ten = lay(ten, True)
                # we need the layer representations just for the original and reconstrcuted, 
                # flatten ,because it's a convolutional shape
                layer_ten = layer_ten.view(len(layer_ten), -1)
            else:
                ten =lay(ten)

        






# %%
