#%%
import torch 
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST

device = torch.device('cuda' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu') 
torch.set_default_device(device)
ds = MNIST(root='data', download=True).data.float().to(device) / 256
train_ds, valid_ds = ds[:50000], ds[50000:]

# %%

