import torch
import torch.nn as nn
from init_helper import init_downconv

class Discriminator(nn.Module):
    def __init__(self, args = None):
        super(Discriminator, self).__init__()
        channel_list = [1,16,32,64,128]
        self.conv = init_downconv(channel_list, 'relu' , 'max')
        self.linear1 = nn.Sequential(nn.Linear(8192, 1000), nn.BatchNorm1d(1000), nn.ReLU())
        self.final = nn.Sequential(nn.Linear(1000,1), nn.Sigmoid())

    def forward(self,x):
        x = self.conv(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.linear1(x)
        x = self.final(x)
        return x

