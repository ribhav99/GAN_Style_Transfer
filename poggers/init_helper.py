import torch
import torch.nn as nn

class SINNode(nn.Module):
    def __init__(self): 
        super(SINNode, self).__init__()

    def forward(self,x):
        return torch.sin(x)

act_fn_module = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()],
                ['sin', SINNode()],
                ['sigmoid', nn.Sigmoid()]
        ])

pool_module = nn.ModuleDict([
                ['max', nn.MaxPool2d(2)],
                ['avg', nn.AvgPool2d(2)],
        ])

def get_norm(channel,norm_type):
    if norm_type == 'instance':
        f = nn.InstanceNorm2d(channel)    
    elif norm_type == 'batch':
        f = nn.BatchNorm2d(channel) 
    else:
        f = nn.Identity()
    return f

def get_down_conv_block(input_channel, output_channel, kersize ,act_fn, norm_type, tonorm = True, stride = 2):
    layer = []
    to_pad = (kersize - 1)//2
    layer.append(nn.Conv2d(input_channel, output_channel,kernel_size =kersize, stride = stride, padding = to_pad))
    if tonorm:
        layer.append(get_norm(output_channel, norm_type))
    layer.append(act_fn_module[act_fn])
    return layer

def get_up_conv_block(input_channel, output_channel, kersize ,act_fn, norm_type, tonorm = True, dropout = False):
    layer = []
    to_pad = (kersize - 1)//2
    layer += [nn.Upsample(scale_factor=2,mode='bilinear', align_corners = False), nn.ReflectionPad2d(1), nn.Conv2d(input_channel,output_channel,kernel_size=3,stride=1,padding=0)]
# layer.append(nn.ConvTranspose2d(input_channel, output_channel,kernel_size =kersize, stride = 2, padding = to_pad))
    if tonorm:
        layer.append(get_norm(output_channel, norm_type))
    if dropout:
        layer.append(nn.Dropout(0.5))
    layer.append(act_fn_module[act_fn])
    return layer

def gen_downconv(channel_list, act_fn, norm_type):
    layer = []
    for i in range(len(channel_list) - 1):
        input_channel = channel_list[i]
        output_channel = channel_list[i+1]
        layer += get_down_conv_block(input_channel,output_channel,4,act_fn,norm_type)
    return layer

def gen_upconv(channel_list, act_fn, norm_type, last_act_fn = 'tanh'):
    channel_list.reverse()
    layer = []
    for i in range(len(channel_list) - 1):
        prev_channel = channel_list[i]
        target_channel = channel_list[i+1]
        if i == (len(channel_list) - 2):
            act_fn = last_act_fn
        layer += get_up_conv_block(prev_channel,target_channel,4,act_fn,norm_type)
    return layer

def get_res_block(in_channel, norm_type, dropout):
    layer = []
    layer.append(get_norm(in_channel,norm_type)) 
    layer.append(nn.ReLU())
    layer.append(nn.Conv2d(in_channel, in_channel, 3, padding =1))
    if dropout:
        layer.append(nn.Dropout(0.5))
    return layer

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, norm_type, dropout = False):
        super(ResidualBlock, self).__init__()
        self.block1 = nn.Sequential(*get_res_block(in_channel, norm_type, dropout))
        self.block2 = nn.Sequential(*get_res_block(in_channel, norm_type, False))

    def forward(self,x):
        residual = x
        output = self.block1(x)
        output = self.block2(output)
        final = residual + output
        return final
