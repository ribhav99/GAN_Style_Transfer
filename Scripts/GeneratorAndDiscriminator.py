from attrdict import AttrDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# size of image after convolution:
# [(W−K+2P)/S]+1
#   W - input dimension
#   K - Kernel Size
#   P - Padding
#   S - Stride

# Minimum image size is 64x64 otherwise max_pooling will crash
# Will need to remove conv layers or remove their max_pooling


class Discriminator(nn.Module):

    def __init__(self, args):
        super().__init__()
        self._linear_dim = None

        # only takes in RGB images
        self.conv1 = nn.Sequential(nn.Conv2d(args.image_dimensions[2], args.features_d, args.kernel_size, stride=2, padding=1), nn.BatchNorm2d(
            args.features_d), nn.ReLU(), nn.MaxPool2d(args.max_pool))
        self.conv2 = nn.Sequential(nn.Conv2d(args.features_d, args.features_d * 2, args.kernel_size, stride=2,
                                             padding=1), nn.BatchNorm2d(args.features_d * 2), nn.ReLU(), nn.MaxPool2d(args.max_pool))
        self.conv3 = nn.Sequential(nn.Conv2d(args.features_d * 2, args.features_d * 4, args.kernel_size,
                                             stride=2, padding=1), nn.BatchNorm2d(args.features_d * 4), nn.ReLU(), nn.MaxPool2d(args.max_pool))

        fake_data = torch.rand(args.image_dimensions).view(
            -1, args.image_dimensions[2], args.image_dimensions[0], args.image_dimensions[1])
        self.conv(fake_data)

        self.linear1 = nn.Sequential(
            nn.Linear(self._linear_dim, 150), nn.BatchNorm1d(150), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(150, 1), nn.Sigmoid())

    def conv(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self._linear_dim is None:
            self._linear_dim = np.prod(x[0].shape)

        return x

    def forward(self, x):
        x = self.conv(x)
        # batch_size = x.shape[0]
        # x = x.view(batch_size, -1)
        # print(x.shape)
        x = x.view(-1, self._linear_dim)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class Generator(nn.Module):

    def __init__(self, args):
        super().__init__()
        self._linear_dim = None

        self.conv1 = nn.Sequential(nn.Conv2d(args.cartoon_dimensions[2], args.features_g, args.kernel_size, padding=1), nn.BatchNorm2d(
            args.features_g), nn.ReLU(), nn.MaxPool2d(args.max_pool))
        self.conv2 = nn.Sequential(nn.Conv2d(args.features_g, args.features_g * 2, args.kernel_size,
                                             padding=1), nn.BatchNorm2d(args.features_g * 2), nn.ReLU(), nn.MaxPool2d(args.max_pool))
        self.conv3 = nn.Sequential(nn.Conv2d(args.features_g * 2, args.features_g * 4, args.kernel_size,
                                             padding=1), nn.BatchNorm2d(args.features_g * 4), nn.ReLU(), nn.MaxPool2d(args.max_pool))

        self.trans_conv1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.ConvTranspose2d(
            args.features_g * 4, args.features_g * 2, args.kernel_size, padding=1), nn.BatchNorm2d(args.features_g * 2), nn.ReLU())
        self.trans_conv2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.ConvTranspose2d(
            args.features_g * 2, args.features_g, args.kernel_size, padding=1), nn.BatchNorm2d(args.features_g), nn.ReLU())
        self.trans_conv3 = nn.Sequential(nn.Upsample(scale_factor=2), nn.ConvTranspose2d(
            args.features_g, args.image_dimensions[2], args.kernel_size, padding=1), nn.Tanh())

    def down_conv(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self._linear_dim is None:
            self._linear_dim = np.prod(x[0].shape)

        return x

    def up_conv(self, x):
        x = self.trans_conv1(x)
        x = self.trans_conv2(x)
        x = self.trans_conv3(x)
        return x

    def forward(self, x):
        x = self.down_conv(x)
        x = self.up_conv(x)
        return x


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# THESE VALUES FOR TESTING. REMOVE THEM

args = AttrDict()
args_dict = {
    'learning_rate': 0.001,
    'batch_size': 1,
    'image_dimensions': (218, 178, 3),
    'cartoon_dimensions': (128, 128, 3),
    'max_pool': (2, 2),
    'features_d': 16,
    'features_g': 16,
    'num_epochs': 30,
    'kernel_size': 3,
    'human_train_path': "/content/GAN_Style_Transfer/data/human_train.txt",
    'human_test_path': "/content/GAN_Style_Transfer/data/human_test.txt",
    'cartoon_train_path': "/content/GAN_Style_Transfer/data/cartoon_train.txt",
    'cartoon_test_path': "/content/GAN_Style_Transfer/data/cartoon_test.txt",
    'human_data_root_path': "/content/humanfaces/",
    'cartoon_data_root_path': "/content/cartoonfaces/",
    'save_path': "/content/GAN_Style_Transfer/Models",
    'image_save_f': 10,  # i.e save an image every 10 epochs
    'use_wandb': False
}
args.update(args_dict)


# x = torch.rand(args.image_dimensions).view(-1, 3,
#                                            args.image_dimensions[0], args.image_dimensions[1])
x = torch.rand(2, 3, 218, 178)

disciminator = Discriminator(args)
print(disciminator(x))

# y = torch.rand(args.cartoon_dimensions).view(-1,
#                                              args.cartoon_dimensions[2], args.cartoon_dimensions[0], args.cartoon_dimensions[1])
y = torch.rand(2, 3, 128, 128)
generator = Generator(args)
y = generator(y)  # .view(-1, 2, args.image_dimensions[0],
# args.image_dimensions[1], args.image_dimensions[2])
print(y.shape)
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
