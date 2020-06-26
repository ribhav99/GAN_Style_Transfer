import torch
import torch.nn as nn

class SINNode(nn.Module):
    def __init__(self): super(SINNode, self).__init__()

    def forward(self,x):
        return torch.sin(x)

act_fn_module = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()],
                ['sin', SINNode()]
        ])

pool_module = nn.ModuleDict([
                ['max', nn.MaxPool2d(2)],
                ['avg', nn.AvgPool2d(2)],
        ])

def get_res_block(in_channel):
    layer = [nn.BatchNorm2d(in_channel), nn.ReLU(), nn.Conv2d(in_channel, in_channel, 3, padding =1)]
    return nn.Sequential(*layer)

def downconv_block(input_channel, output_channel, act_fn, pool_type):
    layer = [nn.Conv2d(input_channel, output_channel, 3, padding=1), nn.BatchNorm2d(output_channel), act_fn_module[act_fn], pool_module[pool_type]]
    return layer

def upconv_block(input_channel, output_channel, act_fn):
    layer = [nn.Upsample(scale_factor=2), nn.ConvTranspose2d(input_channel,output_channel,3,padding=1), nn.BatchNorm2d(output_channel), act_fn_module[act_fn]]
    return layer

def init_downconv(channel_list, act_fn, pool_type):
    layers = []
    for i in range(len(channel_list) - 1):
        layers += downconv_block(channel_list[i], channel_list[i+1], act_fn, pool_type)
    return nn.Sequential(*layers)

def init_upconv(channel_list, act_fn, last_layer_act = 'tanh'):
    channel_list.reverse()
    layers = []
    for i in range(len(channel_list) - 1):
        if i == len(channel_list) - 2:
            act_fn = last_layer_act
        layers += upconv_block(channel_list[i], channel_list[i+1], act_fn)
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()
        self.block1 = get_res_block(in_channel)
        self.block2 = get_res_block(in_channel)

    def forward(self,x):
        residual = x
        output = self.block1(x)
        output = self.block2(output)
        final = residual + output
        return final

