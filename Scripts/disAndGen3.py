from attrdict import AttrDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_res_block(args):
    in_channel = 256
    layer = []
    layer.append(nn.ReflectionPad2d(args.padding))
    layer.append(nn.Conv2d(in_channel, in_channel, 3))
    layer.append(args.norm(in_channel))
    layer.append(args.activation())
    if args.dropout:
        layer.append(nn.Dropout(0.5))
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, args):
        super(ResidualBlock, self).__init__()
        self.block1 = nn.Sequential(
            *get_res_block(args))
        self.block2 = nn.Sequential(
            *get_res_block(args))

    def forward(self, x):
        residual = x
        output = self.block1(x)
        output = self.block2(output)
        final = residual + output
        return final


class Generator(nn.Module):

    def __init__(self, args):
        super().__init__()

        model = []
        model.append(nn.Conv2d(1, 64,
                               kernel_size=7, stride=1))
        model.append(args.norm(64))
        model.append(args.activation())

        model.append(nn.ReflectionPad2d(args.padding + 1))

        model.append(nn.Conv2d(64, 128,
                               kernel_size=3, stride=2))
        model.append(args.norm(128))
        model.append(args.activation())
        model.append(nn.ReflectionPad2d(args.padding + 1))

        model.append(nn.Conv2d(128, 256,
                               kernel_size=3, stride=2))
        model.append(args.norm(256))
        model.append(args.activation())

        for i in range(args.num_residual_layers):
            model += [ResidualBlock(args)]

        model += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.ReflectionPad2d(
            args.padding), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)]

        model.append(args.norm(128))
        model.append(args.activation())

        model += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.ReflectionPad2d(
            args.padding + 1), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)]

        model.append(args.norm(64))
        model.append(args.activation())

        model.append(nn.ReflectionPad2d(args.padding + 1))
        model.append(nn.Conv2d(64, 1,
                               kernel_size=7, stride=1))
        model.append(args.norm(1))
        model.append(args.activation())

        self.go = nn.Sequential(*model)
        # self.initialise_weights()

    def forward(self, x):
        return self.go(x)


class Discriminator(nn.Module):

    def __init__(self, args):
        super().__init__()
        model = []
        model.append(nn.Conv2d(1, 64,
                               kernel_size=4, stride=2))
        model.append(args.activation())

        model.append(nn.Conv2d(64, 128,
                               kernel_size=4, stride=2))
        model.append(args.norm(128))
        model.append(args.activation())

        model.append(nn.Conv2d(128, 256,
                               kernel_size=4, stride=2))
        model.append(args.norm(256))
        model.append(args.activation())

        model.append(nn.Conv2d(256, 512,
                               kernel_size=4, stride=2))
        model.append(args.norm(512))
        model.append(args.activation())

        model.append(nn.Conv2d(512, 1,
                               kernel_size=4, stride=2))
        model.append(args.norm(1))
        model.append(args.activation())

        self.conv = nn.Sequential(*model)

        fake_data = torch.rand(
            2, args.image_dimensions[2], args.image_dimensions[0], args.image_dimensions[1])

        self._linear_dim = np.prod(self.conv(fake_data)[0].shape)
        self.linear = nn.Sequential(
            nn.Linear(self._linear_dim, 1), nn.Sigmoid())
        # self.initialise_weights()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self._linear_dim)
        x = self.linear(x)
        return x


if __name__ == '__main__':

    args = AttrDict()
    args_dict = {
        'dis_learning_rate': 0.0002,
        'gen_learning_rate': 0.0002,
        'image_dimensions': (128, 128, 1),
        'cartoon_dimensions': (128, 128, 1),
        'batch_size': 10,
        'max_pool': (2, 2),
        'num_epochs': 50,
        'kernel_size': 4,
        'padding': 1,  # (kernel_size - 1) //2
        # first entry must match last entry of cartoon dim
        'gen_channels': [1, 32, 64, 128, 256],  # starts 3 for coloured
        'dis_channels': [1, 32, 64, 128, 256],
        'num_residual_layers': 6,
        'image_save_f': 1,
        'discrim_train_f': 1,
        'lambda_cycle': 10,
        'dropout': True,
        'decay': True,
        'load_models': True,
        'model_path': "/content/model100.pt",
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

    discriminator = Discriminator(args)

    x = discriminator(x)
    print(x.shape)

    y = torch.rand(args.batch_size,
                   args.cartoon_dimensions[2], args.cartoon_dimensions[0], args.cartoon_dimensions[1])
    generator = Generator(args)
    y = generator(y)
    print(y.shape)
