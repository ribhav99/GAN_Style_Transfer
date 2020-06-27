import torch
from Generator import Generator
from Discriminator import Discriminator

with torch.no_grad():
    x = torch.randn(10,3,128,128)
    gen = Generator()
    dis = Discriminator()
    y = gen(x)
    z = dis(x)
    print("generator output shape: {}".format(y.shape))
    print("discriminator output shape: {}".format(z.shape))
    print(dis)
    print(gen)
