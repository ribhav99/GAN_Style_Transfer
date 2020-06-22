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
        self.conv1 = nn.Conv2d(args.image_dimensions[2], args.features_d, args.kernel_size, stride=2, padding=1)
        self.conv2 = nn.Conv2d(args.features_d, args.features_d * 2, args.kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(args.features_d * 2, args.features_d * 4, args.kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(args.features_d * 4, args.features_d * 8, args.kernel_size, stride=2, padding=1)
        self.conv5 = nn.Conv2d(args.features_d * 8, 1, args.kernel_size, stride=2, padding=1)

        fake_data = torch.rand(args.image_dimensions).view(-1, args.image_dimensions[2], args.image_dimensions[0], args.image_dimensions[1])
        self.conv(fake_data)
        
        self.dense1 = nn.Linear(self._linear_dim, 512)
        self.dense2 = nn.Linear(512, 1)
    
    def conv(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), args.max_pool)
        x = F.max_pool2d(F.relu(self.conv2(x)), args.max_pool)
        x = F.max_pool2d(F.relu(self.conv3(x)), args.max_pool)
        if min(args.image_dimensions) >= 256:
            x = F.max_pool2d(F.relu(self.conv4(x)), args.max_pool)
        if min(args.image_dimensions) >= 1024:
            x = F.max_pool2d(F.relu(self.conv5(x)), args.max_pool)

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

    def __init__(self, args):
        super().__init__()
        self._linear_dim = None

        self.conv1 = nn.Conv2d(args.cartoon_dimensions[2], args.features_g, args.kernel_size, padding=1)
        self.conv2 = nn.Conv2d(args.features_g, args.features_g * 2, args.kernel_size, padding=1)
        self.conv3 = nn.Conv2d(args.features_g * 2, args.features_g * 4, args.kernel_size, padding=1)

        fake_data = torch.rand(args.cartoon_dimensions).view(-1, args.cartoon_dimensions[2], args.cartoon_dimensions[0], args.cartoon_dimensions[1])
        self.down_conv(fake_data)

        self.dense1 = nn.Linear(self._linear_dim, args.image_dimensions[0] * args.image_dimensions[1] * args.image_dimensions[2])

        self.trans_conv1 = nn.ConvTranspose2d(args.image_dimensions[2], args.features_g * 8, args.kernel_size, stride=2, padding=0)
        self.trans_conv2 = nn.ConvTranspose2d(args.features_g * 8, args.features_g * 4, args.kernel_size, stride=2, padding=0)
        self.trans_conv3 = nn.ConvTranspose2d(args.features_g * 4, args.image_dimensions[2], args.kernel_size, stride=2, padding=0)
    
    def down_conv(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), args.max_pool)
        x = F.max_pool2d(F.relu(self.conv2(x)), args.max_pool)
        x = F.max_pool2d(F.relu(self.conv3(x)), args.max_pool)

        if self._linear_dim is None:
            self._linear_dim = np.prod(x[0].shape)
        
        return x

    def up_conv(self, x):
        x = F.max_pool2d(F.relu(self.trans_conv1(x)), args.max_pool)
        x = F.max_pool2d(F.relu(self.trans_conv2(x)), args.max_pool)
        x = F.max_pool2d(F.relu(self.trans_conv3(x)), args.max_pool)
        
        return x
    
    def forward(self, x):
        x = self.down_conv(x)
        x = x.view(-1, self._linear_dim)
        x = F.relu(self.dense1(x))
        x = x.view(-1, args.image_dimensions[2], args.image_dimensions[0], args.image_dimensions[1])
        x = self.up_conv(x)
        return x


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# THESE VALUES FOR TESTING. REMOVE THEM

# from attrdict import AttrDict
# args = AttrDict()
# args_dict = {
# 'learning_rate': 0.001,
# 'batch_size' : 64,
# 'image_dimensions' : (218, 178, 3), 
# 'cartoon_dimensions' : (128, 128, 3),
# 'max_pool' : (2, 2),
# 'features_d' : 2,
# 'features_g' : 2,
# 'num_epochs' : 30,
# 'kernel_size' : 3,
# 'human_train_path' : "/content/GAN_Style_Transfer/data/human_train.txt",
# 'human_test_path' : "/content/GAN_Style_Transfer/data/human_test.txt",
# 'cartoon_train_path' : "/content/GAN_Style_Transfer/data/cartoon_train.txt",
# 'cartoon_test_path' : "/content/GAN_Style_Transfer/data/cartoon_test.txt",
# 'human_data_root_path' : "/content/humanfaces/",
# 'cartoon_data_root_path' : "/content/cartoonfaces/",
# 'save_path' : "/content/GAN_Style_Transfer/Models",
# 'image_save_f' : 10, #i.e save an image every 10 epochs
# 'use_wandb' : False
# }
# args.update(args_dict)


# x = torch.rand(args.image_dimensions).view(-1, 3, args.image_dimensions[0], args.image_dimensions[1])

# disciminator = Discriminator(args)
# print(disciminator(x))

# y = torch.rand(args.cartoon_dimensions).view(-1, args.cartoon_dimensions[2], args.cartoon_dimensions[0], args.cartoon_dimensions[1])
# generator = Generator(args)
# y = generator(y).view(-1, args.image_dimensions[0], args.image_dimensions[1], args.image_dimensions[2])
# print(y.shape)
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX