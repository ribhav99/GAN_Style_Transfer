import torch
from Generator import Generator
from Discriminator import Discriminator

def test(args):
    with torch.no_grad():
        x = torch.randn(10,3,128,128)
        gen = Generator(args)
        dis = Discriminator(args)
        y = gen(x)
        z = dis(x)
        print("generator output shape: {}".format(y.shape))
        print("discriminator output shape: {}".format(z.shape))
        print(dis)
        print(gen)

if __name__ == "__main__":
    from attrdict import AttrDict
    args = AttrDict()
    args_dict = {
    'norm_type' :'instance',
    'act_fn_gen' : 'relu',
    'act_fn_dis' : 'lrelu',
    'num_res' : 3
    }
    args.update(args_dict)
    test(args)
