# %% 
# torch.utils.tensorboard
# https://pytorch.org/docs/stable/tensorboard.html

# %%
from numpy import random
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# writer will output to ./runs/ directory by default
writer = SummaryWriter()

# %%
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5,))])
trainset = datasets.MNIST(
    r"H:/ComputerVision/PyTorch/GAN_implementatnion/data/",
    train=True,
    download=False,
    transform=transform,
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle=True)
model = torchvision.models.resnet50(False)

model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
images, labels = next(iter(trainloader))

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()
# %%
import numpy as np 

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

# %%
