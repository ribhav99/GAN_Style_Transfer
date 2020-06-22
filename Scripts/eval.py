import torch
import numpy as np
from GeneratorAndDiscriminator import Generator, Discriminator
import skimage.io as io
import matplotlib.pyplot as plt
def getImage(pathname):
    image = io.imread(pathname) 
    image = np.moveaxis(image,-1,0)
    return torch.from_numpy(image).float()

def plot_img(image):
    image = np.moveaxis(image,0,-1)
    plt.imshow(image)
    plt.show()
    io.imsave("/Users/gerald/Desktop/results/sampleimage.png", image)

def getArgs():
    root_human_data = "/Users/gerald/Desktop/GAN datasets/humanfaces"
    root_cartoon_data = "/Users/gerald/Desktop/GAN datasets/cartoonfaces"
    ####
    from attrdict import AttrDict
    args = AttrDict()
    args_dict = {
    'learning_rate': 0.001,
    'batch_size' : 64,
    'image_dimensions' : (218, 178, 3),
    'cartoon_dimensions' : (128, 128, 3),
    'max_pool' : (2, 2),
    'features_d' : 2,
    'features_g' : 2,
    'num_epochs' : 30,
    'kernel_size' : 3,
    'human_train_path' : "/content/GAN_Style_Transfer/data/human_train.txt",
    'human_test_path' : "/content/GAN_Style_Transfer/data/human_test.txt",
    'cartoon_train_path' : "/content/GAN_Style_Transfer/data/cartoon_train.txt",
    'cartoon_test_path' : "/content/GAN_Style_Transfer/data/cartoon_test.txt",
    'human_data_root_path' : "/content/humanfaces/",
    'cartoon_data_root_path' : "/content/cartoonfaces/",
    'save_path' : "/content/GAN_Style_Transfer/Models",
    'image_save_f' : 10, #i.e save an image every 10 epochs
    'use_wandb' : False
    }
    args.update(args_dict)
    return args
if __name__ == "__main__":
    gen_path = "/Users/gerald/Desktop/generator.pt"
    device = torch.device('cpu')
    x = getImage("/Users/gerald/Desktop/GAN datasets/cartoonfaces/cs7165328004066367879.png")
    args = getArgs()
    model = Generator(args.cartoon_dimensions, args.image_dimensions, features_g=args.features_g, kernel_size=args.kernel_size, max_pool=args.max_pool)
    model.load_state_dict(torch.load(gen_path, map_location=device))
    model.eval()
    with torch.no_grad():
        output = model(x.unsqueeze(0))
        output = output.squeeze(0).numpy()
    plot_img(output) 
