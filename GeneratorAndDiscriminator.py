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

    def __init__(self, features_d, kernel_size):
        super().__init__()
        self._linear_dim = None

        # only takes in RGB images
        self.conv1 = nn.Conv2d(3, features_d, kernel_size, stride=2, padding=1)
        self.conv2 = nn.Conv2d(features_d, features_d * 2, kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(features_d * 2, features_d * 4, kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(features_d * 4, features_d * 8, kernel_size, stride=2, padding=1)
        self.conv5 = nn.Conv2d(features_d * 8, 1, kernel_size, stride=2, padding=1)

        fake_data = torch.rand(image_dimensions).view(-1, image_dimensions[2], image_dimensions[0], image_dimensions[1])
        self.conv(fake_data)
        
        self.dense1 = nn.Linear(self._linear_dim, 512)
        self.dense2 = nn.Linear(512, 1)
    
    def conv(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), max_pool)
        x = F.max_pool2d(F.relu(self.conv2(x)), max_pool)
        x = F.max_pool2d(F.relu(self.conv3(x)), max_pool)
        if min(image_dimensions) >= 256:
            x = F.max_pool2d(F.relu(self.conv4(x)), max_pool)
        if min(image_dimensions) >= 1024:
            x = F.max_pool2d(F.relu(self.conv5(x)), max_pool)

        if self._linear_dim is None:
            self._linear_dim = np.prod(x[0].shape)

        return x

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self._linear_dim)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        x = torch.sigmoid(x)
        return x



class Generator(nn.Module):

    def __init__(self, features_g, kernel_size):
        super().__init__()
        self._linear_dim = None

        #takes in grayscale images
        self.conv1 = nn.ConvTranspose2d(3, features_g, kernel_size, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(features_g, features_g * 16, kernel_size, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(features_g * 16, features_g * 8, kernel_size, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size, stride=2, padding=1)
        self.conv6 = nn.ConvTranspose2d(features_g * 2, features_g, kernel_size, stride=2, padding=1)


        fake_data = torch.rand(cartoon_dimensions).view(-1, cartoon_dimensions[2], cartoon_dimensions[0], cartoon_dimensions[1])
        self.conv(fake_data)

        self.dense1 = nn.Linear(self._linear_dim, (image_dimensions[0] * image_dimensions[1] * image_dimensions[2])//2)
        self.dense2 = nn.Linear((image_dimensions[0] * image_dimensions[1] * image_dimensions[2])//2, image_dimensions[0] * image_dimensions[1] * image_dimensions[2])
    
    def conv(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), max_pool)
        x = F.max_pool2d(F.relu(self.conv2(x)), max_pool)
        x = F.max_pool2d(F.relu(self.conv3(x)), max_pool)
        x = F.max_pool2d(F.relu(self.conv4(x)), max_pool)
        x = F.max_pool2d(F.relu(self.conv5(x)), max_pool)
        x = F.max_pool2d(F.relu(self.conv6(x)), max_pool)

        if self._linear_dim is None:
            self._linear_dim = np.prod(x[0].shape)
        
        return x
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self._linear_dim)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return x

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#THESE VALUES FOR TESTING. REMOVE THEM
features_d = 64
kernel_size = 3
image_dimensions = (128, 128, 3)
max_pool = (2,2)
x = torch.rand(image_dimensions).view(-1, 3, image_dimensions[0], image_dimensions[1])

disciminator = Discriminator(features_d, kernel_size)
print(disciminator(x))


features_g = 2
cartoon_dimensions = (128, 128, 3)
y = torch.rand(cartoon_dimensions).view(-1, cartoon_dimensions[2], cartoon_dimensions[0], cartoon_dimensions[1])
generator = Generator(features_g, kernel_size)
y = generator(y).view(-1, image_dimensions[0], image_dimensions[1], image_dimensions[2])
print(y.shape)
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX