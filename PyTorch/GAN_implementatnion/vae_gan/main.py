# %%
import torch
import numpy
import argparse

numpy.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)

from network import VaeGan
from torch.autograd import Variable
from torch.utils.data import Dataset, dataset
from tensorboardX import SummaryWriter
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
import progressbar
from torchvision.utils import make_grid
from generator import CELEBA, CELEBA_SLURM
# from utils import RollingMeasure

# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VAEGAN")
    parser.add_argument("--train_folder",action="store",dest="train_folder")
    parser.add_argument("--test_folder",action="store",dest="test_folder")
    parser.add_argument("--n_epochs",default=12,action="store",type=int,dest="n_epochs")
    parser.add_argument("--z_size",default=128,action="store",type=int,dest="z_size")
    parser.add_argument("--recon_level",default=3,action="store",type=int,dest="recon_level")
    parser.add_argument("--lambda_mse",default=1e-6,action="store",type=float,dest="lambda_mse")
    parser.add_argument("--lr",default=3e-4,action="store",type=float,dest="lr")
    parser.add_argument("--decay_lr",default=0.75,action="store",type=float,dest="decay_lr")
    parser.add_argument("--decay_mse",default=1,action="store",type=float,dest="decay_mse")
    parser.add_argument("--decay_margin",default=1,action="store",type=float,dest="decay_margin")
    parser.add_argument("--decay_equilibrium",default=1,action="store",type=float,dest="decay_equilibrium")
    parser.add_argument("--slurm",default=False,action="store",type=bool,dest="slurm")

    args = parser.parse_args([])

    train_folder = args.train_folder
    test_folder = args.test_folder
    z_size = args.z_size
    recon_level = args.recon_level
    decay_mse = args.decay_mse
    decay_margin = args.decay_margin
    n_epochs = args.n_epochs
    lambda_mse = args.lambda_mse
    lr = args.lr
    decay_lr = args.decay_lr
    decay_equilibrium = args.decay_equilibrium
    slurm = args.slurm
    
    writer = SummaryWriter(comment="_CELEBA_NEW_DATA_STOCK_GAN")
    net = VaeGan(z_size=z_size, recon_level=recon_level).cuda()

    # dataset
    if not slurm:
        dataloader = Dataset.DataLoader(CELEBA(train_folder), batch_size=64,
                                                shuffle=True, num_workers=4)
        # dataset for test
        # if you want to split train from test just move some files in another dir
        dataloader_test = Dataset.DataLoader(CELEBA(test_folder), batch_size=100,
                                                shuffle=False, num_workers=1)
    else:
        dataloader = Dataset.DataLoader(CELEBA_SLURM(train_folder), bath_size=64,
                                                shuffle=True, num_workers=1)
        # dataset for test 
        # if you want to split train from test just move some files in another dir
        dataloader_test = Dataset.DataLoader(CELEBA_SLURM(test_folder), batch_size=100,
                                                shuffle=False, num_workers=1)

    # margin and equilibirum 
    margin = 0.35
    equilibirum = 0.68

    # optim-loss
    # an optimizer for each of the sub-networks, so we can selectively backprop
    optimizer_encoder = RMSprop(params=net.encoder.parameters(), lr=lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    lr_encoder = ExponentialLR(optimizer_encoder, gamma=decay_lr)

    optimizer_decoder = RMSprop(params=net.decoder.parameters(), lr=lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    lr_decoder = ExponentialLR(optimizer_decoder, gamma=decay_lr)

    optimizer_discriminator = RMSprop(params=net.discriminator.parameters(), lr=lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    lr_discriminator = ExponentialLR(optimizer_discriminator, gamma=decay_lr)

    batch_number = len(dataloader)
    step_index = 0
    




