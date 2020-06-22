import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, args = None):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding = 1),nn.BatchNorm2d(16) ,nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding = 1),nn.BatchNorm2d(32) ,nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding = 1),nn.BatchNorm2d(64) ,nn.ReLU(), nn.MaxPool2d(2))
        self.linear1 = nn.Sequential(nn.Linear(256, 150), nn.BatchNorm1d(150), nn.ReLU())
        self.final = nn.Sequential(nn.Linear(150,1), nn.Sigmoid())

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.linear1(x)
        x = self.final(x)
        return x
