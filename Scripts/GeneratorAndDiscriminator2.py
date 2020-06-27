from attrdict import AttrDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_res_block(args, dropout):
    in_channel = args.gen_channels[-1]
    layer = []
    layer.append(args.norm(in_channel))
    layer.append(args.activation())
    layer.append(nn.Conv2d(in_channel, in_channel, 3, padding=1))
    if dropout:
        layer.append(nn.Dropout(0.5))
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, args, dropout=False):
        super(ResidualBlock, self).__init__()
        self.block1 = nn.Sequential(
            *get_res_block(args, dropout))
        self.block2 = nn.Sequential(
            *get_res_block(args, dropout))

    def forward(self, x):
        residual = x
        output = self.block1(x)
        output = self.block2(output)
        final = residual + output
        return final


class Generator(nn.Module):

    def __init__(self, args):
        super().__init__()

        model = self.gen_downconv(args)

        for i in range(args.num_residual_layers):
            model += [ResidualBlock(args)]

        model += self.gen_upconv(args)

        self.go = nn.Sequential(*model)

    def forward(self, x):
        return self.go(x)

    def get_down_conv_block(self, input_channel, output_channel, args, stride=2):
        layer = []
        layer.append(nn.Conv2d(input_channel, output_channel,
                               kernel_size=args.kernel_size, stride=stride, padding=args.padding))
        layer.append(args.norm(output_channel))
        layer.append(args.activation())
        return layer

    def gen_downconv(self, args):
        layer = []
        for i in range(len(args.gen_channels) - 1):
            input_channel = args.gen_channels[i]
            output_channel = args.gen_channels[i+1]
            layer += self.get_down_conv_block(input_channel,
                                              output_channel, args)

        return layer

    def gen_upconv(self, args, last_act_fn='tanh'):
        channel_list = list(args.gen_channels)
        channel_list.reverse()
        layer = []
        for i in range(len(channel_list) - 1):
            prev_channel = channel_list[i]
            target_channel = channel_list[i+1]

            if i == (len(channel_list) - 2):
                act_fn = True
            else:
                act_fn = False

            layer += self.get_up_conv_block(prev_channel,
                                            target_channel, args, act_fn)
        return layer

    def get_up_conv_block(self, input_channel, output_channel, args, act_fn, dropout=False):
        layer = []
        layer.append(nn.ConvTranspose2d(input_channel, output_channel,
                                        kernel_size=args.kernel_size, stride=2, padding=args.padding))

        layer.append(args.norm(output_channel))
        if dropout:
            layer.append(nn.Dropout(0.5))
        if act_fn:
            layer.append(nn.Tanh())
        else:
            layer.append(args.activation())
        return layer


class Discriminator(nn.Module):

    def __init__(self, args):
        super().__init__()
        conv_layers = self.dis_downconv(args)
        conv_layers.append(nn.Conv2d(args.dis_channels[-1], 1,
                                     kernel_size=1, stride=2, padding=0))
        conv_layers.append(args.activation())
        self.conv = nn.Sequential(*conv_layers)

        fake_data = torch.rand(
            2, args.image_dimensions[2], args.image_dimensions[0], args.image_dimensions[1])

        self._linear_dim = np.prod(self.conv(fake_data)[0].shape)
        print(self._linear_dim)
        self.linear = nn.Sequential(
            nn.Linear(self._linear_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self._linear_dim)
        x = self.linear(x)
        return x

    def get_down_conv_block(self, input_channel, output_channel, args, stride=2):
        layer = []
        layer.append(nn.Conv2d(input_channel, output_channel,
                               kernel_size=args.kernel_size, stride=stride, padding=args.padding))
        layer.append(args.norm(output_channel))
        layer.append(args.activation())
        return layer

    def dis_downconv(self, args):
        layer = []
        for i in range(len(args.dis_channels) - 1):
            input_channel = args.dis_channels[i]
            output_channel = args.dis_channels[i+1]
            layer += self.get_down_conv_block(input_channel,
                                              output_channel, args)
        return layer


if __name__ == '__main__':

    args = AttrDict()
    args_dict = {
        'dis_learning_rate': 0.001,
        'gen_learning_rate': 0.002,
        'image_dimensions': (128, 128, 1),
        'cartoon_dimensions': (128, 128, 1),
        'batch_size': 2,
        'max_pool': (2, 2),
        'num_epochs': 85,
        'kernel_size': 4,
        'padding': 1,  # (kernel_size - 1) //2
        # first entry must match last entry of cartoon dim
        'gen_channels': [1, 32, 64, 128, 256],
        'dis_channels': [1, 32, 64, 128, 256],
        'num_residual_layers': 3,
        'image_save_f': 1,
        'discrim_train_f': 3,
        'discrim_error_train': False,
        'pool': nn.AvgPool2d,
        'activation': nn.LeakyReLU,
        'norm': nn.InstanceNorm2d,
        'human_train_path': "/content/GAN_Style_Transfer/data/human_train.txt",
        'human_test_path': "/content/GAN_Style_Transfer/data/human_test.txt",
        'cartoon_train_path': "/content/GAN_Style_Transfer/data/cartoon_train.txt",
        'cartoon_test_path': "/content/GAN_Style_Transfer/data/cartoon_test.txt",
        'human_data_root_path': "/content/humangray128/",
        'cartoon_data_root_path': "/content/cartoonfacesgray/",
        'save_path': "/content/GAN_Style_Transfer/Models",
        'use_wandb': True
    }
    args.update(args_dict)

    x = torch.rand(args.batch_size,
                   args.image_dimensions[2], args.image_dimensions[0], args.image_dimensions[1])

    disciminator = Discriminator(args)
    x = disciminator(x)
    print(x.shape)

    y = torch.rand(args.batch_size,
                   args.cartoon_dimensions[2], args.cartoon_dimensions[0], args.cartoon_dimensions[1])
    generator = Generator(args)
    y = generator(y)
    print(y.shape)
