import torch
import matplotlib.pyplot as plt
import numpy as np
from Generator import Generator
import skimage.io as io
import os
from attrdict import AttrDict

def get_args():
    args = AttrDict()
    args_dict = {
      'dis_learning_rate': 0.0002,
      'gen_learning_rate': 0.0002,
      'batch_size' : 200,
      'num_epochs' : 40,
      'human_train_path' : "/content/GAN_Style_Transfer/data/human_train.txt",
      'human_test_path' : "/content/GAN_Style_Transfer/data/human_test.txt",
      'cartoon_train_path' : "/content/GAN_Style_Transfer/data/cartoon_train.txt",
      'cartoon_test_path' : "/content/GAN_Style_Transfer/data/cartoon_test.txt",
      'image_save_f' : 1,
      'act_fn_gen': 'relu',
      'act_fn_dis': 'lrelu',
      'norm_type' :'batch',
      'num_res' : 3,
      'lambda_cycle': 10,
      'use_wandb' : True
    }
    args.update(args_dict)
    return args

def get_images(root_dir, num):
    x = os.listdir(root_dir)
    im_names = np.random.choice(x, size = num, replace = False)
    im_list = []
    for item in im_names:
        if item.endswith(".jpg") or item.endswith(".png"):
            image = io.imread(root_dir + "/" + item)
            image = np.moveaxis(image,-1,0)
            im_list.append(image)
    return torch.tensor(im_list).float(), im_names

def load_gen(gen_path):
    device = torch.device('cpu')
    model = Generator(get_args())
    model.load_state_dict(torch.load(gen_path, map_location=device))
    model.eval()
    return model

def save_img(image, save_name):
    image = (image + 1)*127.5
    io.imsave(save_name, image)

def main(gen_path_x_y, gen_path_y_x, root_dir_x, root_dir_y, target_dir, num):
    x_images, x_im_names = get_images(root_dir_x, num)
    y_images, y_im_names = get_images(root_dir_y, num)
    g_x_y = load_gen(gen_path_x_y)
    g_y_x = load_gen(gen_path_y_x)
    with torch.no_grad():
        fake_y = g_x_y(y_images).numpy()
        fake_x = g_y_x(x_images).numpy()
        for i in range(num):
            image = (fake_x[i] + 1)/2
            image = np.moveaxis(image,0,-1)
            plt.imshow(image)
            plt.show()
# save_img(fake_y[i], target_dir + "/" + "fakehumansample" + str(i) + ".png")
# save_img(fake_x[i], target_dir + "/" + "fakecartoonsample" + str(i) + ".png")

if __name__ == "__main__":
    gen_path_x_y = "/Users/gerald/Desktop/models/g_x_y.pt"
    gen_path_y_x = "/Users/gerald/Desktop/models/g_y_x.pt"
    root_dir_y = "/Users/gerald/Desktop/GAN datasets/humanfaces128"
    root_dir_x = "/Users/gerald/Desktop/GAN datasets/cartoonfaces"
    target_dir = "/Users/gerald/Desktop/models/results"
    main(gen_path_x_y,gen_path_y_x,root_dir_x, root_dir_y, target_dir,10)
