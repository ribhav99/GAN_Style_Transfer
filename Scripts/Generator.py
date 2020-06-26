import torch
import torch.nn as nn
from init_helper import ResidualBlock, init_upconv, init_downconv

class Generator(nn.Module):
    def __init__(self, args=None):
        super(Generator, self).__init__()
        channel_list = [1,16,32,64]
        act_fn = 'relu' if args is None else args.act_fn
        pool_type = 'max' if args is None else args.pool_type
        self.downconv = init_downconv(channel_list, act_fn, pool_type)
        self.resblock = ResidualBlock(64)
        self.upconv = init_upconv(channel_list, act_fn, last_layer_act = 'tanh')

    def forward(self, x):
        x = self.downconv(x)
        x = self.resblock(x)
        x = self.upconv(x)
        return x

