import torch
import torch.nn as nn
from init_helper import gen_downconv, get_down_conv_block

class Discriminator(nn.Module):
    def __init__(self, args = None):
        super(Discriminator, self).__init__()
        channel_list = [3,16,32,64,128,256]
        act_fn = 'relu' if args is None else args.act_fn
        norm_type = 'batch' if args is None else args.norm_type
        conv_layers = gen_downconv(channel_list,act_fn,norm_type)
        conv_layers += get_down_conv_block(256,1,1,act_fn,norm_type,False,1)
        self.conv = nn.Sequential(*conv_layers)
        self.linear = nn.Sequential(nn.Linear(16,1), nn.Sigmoid())

    def forward(self,x):
        x = self.conv(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x

