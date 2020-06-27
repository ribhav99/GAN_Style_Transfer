import torch
import torch.nn as nn
from init_helper import ResidualBlock, gen_upconv, gen_downconv


class Generator(nn.Module):
    def __init__(self, args=None):
        super(Generator, self).__init__()
        act_fn = 'relu' if args is None else args.act_fn
        norm_type = 'batch' if args is None else args.norm_type
        num_res = 3 if args is None else args.num_res
        channel_list = [3,32,64,128, 256]
        model = gen_downconv(channel_list, act_fn,norm_type)
        for i in range(num_res):
            model += [ResidualBlock(256, norm_type)]
        model += gen_upconv(channel_list,act_fn,norm_type)
        self.go = nn.Sequential(*model)
    def forward(self, x):
        x = self.go(x)
        return x

