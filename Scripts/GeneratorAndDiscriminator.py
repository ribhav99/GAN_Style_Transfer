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

    def __init__(self, input_dimensions, features_d=4, kernel_size=3, max_pool=(2,2)):
        super().__init__()
        self.input_dimensions = input_dimensions
        self.features_d = features_d
        self.kernel_size = kernel_size
        self.max_pool = max_pool
        self._linear_dim = None

        # only takes in RGB images
        self.conv1 = nn.Conv2d(self.input_dimensions[2], self.features_d, self.kernel_size, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.features_d, self.features_d * 2, self.kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(self.features_d * 2, self.features_d * 4, self.kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(self.features_d * 4, self.features_d * 8, self.kernel_size, stride=2, padding=1)
        self.conv5 = nn.Conv2d(self.features_d * 8, 1, self.kernel_size, stride=2, padding=1)

        fake_data = torch.rand(self.input_dimensions).view(-1, self.input_dimensions[2], self.input_dimensions[0], self.input_dimensions[1])
        self.conv(fake_data)
        
        self.dense1 = nn.Linear(self._linear_dim, 512)
        self.dense2 = nn.Linear(512, 1)
    
    def conv(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), self.max_pool)
        x = F.max_pool2d(F.relu(self.conv2(x)), self.max_pool)
        x = F.max_pool2d(F.relu(self.conv3(x)), self.max_pool)
        if min(self.input_dimensions) >= 256:
            x = F.max_pool2d(F.relu(self.conv4(x)), self.max_pool)
        if min(self.input_dimensions) >= 1024:
            x = F.max_pool2d(F.relu(self.conv5(x)), self.max_pool)

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

    def __init__(self, input_dimensions, output_dimensions, features_g=4, kernel_size=3, max_pool=(2,2)):
        super().__init__()
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.features_g = features_g
        self.kernel_size = kernel_size
        self.max_pool = max_pool
        self._linear_dim = None

        self.conv1 = nn.Conv2d(self.input_dimensions[2], self.features_g, self.kernel_size, padding=1)
        self.conv2 = nn.Conv2d(features_g, features_g * 2, kernel_size, padding=1)

        self.trans_conv1 = nn.ConvTranspose2d(self.features_g * 2, self.features_g * 16, self.kernel_size, stride=2, padding=1)
        self.trans_conv2 = nn.ConvTranspose2d(self.features_g * 16, self.features_g * 8, self.kernel_size, stride=2, padding=1)
        self.trans_conv3 = nn.ConvTranspose2d(self.features_g * 8, self.features_g * 4, self.kernel_size, stride=2, padding=1)
        self.trans_conv4 = nn.ConvTranspose2d(self.features_g * 4, self.features_g * 2, self.kernel_size, stride=2, padding=1)
        self.trans_conv5 = nn.ConvTranspose2d(self.features_g * 2, self.features_g, self.kernel_size, stride=2, padding=1)


        fake_data = torch.rand(self.input_dimensions).view(-1, self.input_dimensions[2], self.input_dimensions[0], self.input_dimensions[1])
        self.conv(fake_data)

        self.dense1 = nn.Linear(self._linear_dim, (self.output_dimensions[0] * self.output_dimensions[1] * self.output_dimensions[2])//2)
        self.dense2 = nn.Linear((self.output_dimensions[0] * self.output_dimensions[1] * self.output_dimensions[2])//2, self.output_dimensions[0] * self.output_dimensions[1] * self.output_dimensions[2])
    
    def conv(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), self.max_pool)
        x = F.max_pool2d(F.relu(self.conv2(x)), self.max_pool)
        x = F.max_pool2d(F.relu(self.trans_conv1(x)), self.max_pool)
        x = F.max_pool2d(F.relu(self.trans_conv2(x)), self.max_pool)
        x = F.max_pool2d(F.relu(self.trans_conv3(x)), self.max_pool)
        x = F.max_pool2d(F.relu(self.trans_conv4(x)), self.max_pool)
        x = F.max_pool2d(F.relu(self.trans_conv5(x)), self.max_pool)

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
# THESE VALUES FOR TESTING. REMOVE THEM
features_d = 64
kernel_size = 3
image_dimensions = (218, 178, 3)
max_pool = (2,2)
x = torch.rand(image_dimensions).view(-1, 3, image_dimensions[0], image_dimensions[1])

disciminator = Discriminator(image_dimensions, features_d, kernel_size, max_pool)
print(disciminator(x))


features_g = 2
cartoon_dimensions = (128, 128, 3)
y = torch.rand(cartoon_dimensions).view(-1, cartoon_dimensions[2], cartoon_dimensions[0], cartoon_dimensions[1])
generator = Generator(cartoon_dimensions, image_dimensions, features_g, kernel_size, max_pool)
y = generator(y).view(-1, image_dimensions[0], image_dimensions[1], image_dimensions[2])
print(y.shape)
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
