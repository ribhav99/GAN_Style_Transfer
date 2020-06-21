import torch.nn as nn
import torch

#face 218,178,3
#cartoon 128 x 128 x4 
class SIRENGenerator(nn.Module):
    def __init__(self):
        super(SIRENGenerator, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, 3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64,1,1))
        self.layer5 = nn.Sequential(nn.Linear(2091, 1000))
        self.layer6 = nn.Sequential(nn.Linear(1000, 2000))
        self.layer7 = nn.Sequential(nn.Linear(2000, 3600))
        self.layer8 = nn.Sequential(nn.ConvTranspose2d(1, 2, 3))
        self.layer9 = nn.Sequential(nn.ConvTranspose2d(2, 4, 3))
        self.layer10 = nn.Upsample(scale_factor =2)

    def forward(self, x):
        N_batch = x.shape[0]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x).view(N_batch, -1)
        x = self.layer5(x)
        x = torch.sin(x)
        x = self.layer6(x)
        x = torch.sin(x)
        x = self.layer7(x)
        x = x.view(N_batch, 1, 60, 60)
        x = self.layer8(x)
        x = torch.sin(x)
        x = self.layer9(x)
        x = torch.sin(x)
        x = self.layer10(x)
        return x 
# x = torch.randn(1,3,218,178)
# gen = SIRENGenerator()
# y = gen(x)
# print(y.shape)
