import torch
import torch.nn as nn
from init_helper import ResidualBlock, gen_upconv, gen_downconv


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        act_fn = args.act_fn_gen
        norm_type = args.norm_type
        num_res = args.num_res
        channel_list = [3,32,64,128, 256]
        if args.gray:
            channel_list[0] = 1
        model = gen_downconv(channel_list, act_fn,norm_type)
        for i in range(num_res):
            model += [ResidualBlock(256, norm_type)]
        model += gen_upconv(channel_list,act_fn,norm_type, dropout = args.dropout, conv2T = args.Conv2T)
        self.go = nn.Sequential(*model)

    def forward(self, x):
        x = self.go(x)
        return x


