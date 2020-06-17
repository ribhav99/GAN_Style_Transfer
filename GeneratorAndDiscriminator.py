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

# Minimum image size is 256x256 otherwise max_pooling will crash
# Will need to remove conv layers or remove their max_pooling

class Discriminator(nn.Module):

    def __init__(self, features_d, kernel_size):
        super().__init__()
        self._linear_dim = None

        # only takes in grayscale images
        self.conv1 = nn.Conv2d(1, features_d, kernel_size, stride=2, padding=1)
        self.conv2 = nn.Conv2d(features_d, features_d * 2, kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(features_d * 2, features_d * 4, kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(features_d * 4, features_d * 8, kernel_size, stride=2, padding=1)
        self.conv5 = nn.Conv2d(features_d * 8, 1, kernel_size, stride=2, padding=1)

        x = torch.rand(image_dimensions).view(-1, 1, image_dimensions[0], image_dimensions[1])
        self.conv(x)
        
        self.dense1 = nn.Linear(self._linear_dim, 512)
        self.dense2 = nn.Linear(512, 1)
    
    def conv(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), max_pool)
        x = F.max_pool2d(F.relu(self.conv2(x)), max_pool)
        x = F.max_pool2d(F.relu(self.conv3(x)), max_pool)
        x = F.max_pool2d(F.relu(self.conv4(x)), max_pool)
        x = F.relu(self.conv5(x))

        if self._linear_dim is None:
            self._linear_dim = np.prod(x[0].shape)

        return x

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self._linear_dim)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = torch.sigmoid(x)
        return x

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#THESE VALUES FOR TESTING. REMOVE THEM
features_d = 64
kernel_size = 3
image_dimensions = (256, 256)
max_pool = (2,2)
x = torch.rand(image_dimensions).view(-1, 1, image_dimensions[0], image_dimensions[1])

disciminator = Discriminator(features_d, kernel_size)
print(disciminator(x))
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX