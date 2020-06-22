import torch
import torch.nn as nn


def init_downconv(input_channel, second_channel, num_layers):
    layers = []
    for i in range(3):
        this_layer = [nn.Conv2d(input_channel, second_channel, 3, padding=1), nn.BatchNorm2d(
            second_channel), nn.ReLU(), nn.MaxPool2d(2)]
        layers += this_layer
        input_channel = second_channel
        second_channel = second_channel * 2
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, args=None):
        super(Generator, self).__init__()
        self.downconv = init_downconv(3, 16, 3)
        self.upconv1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.ConvTranspose2d(
            64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.upconv2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.ConvTranspose2d(
            32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.upconv3 = nn.Sequential(nn.Upsample(
            scale_factor=2), nn.ConvTranspose2d(16, 3, 3, padding=1), nn.Tanh())

    def forward(self, x):
        x = self.downconv(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        return x


