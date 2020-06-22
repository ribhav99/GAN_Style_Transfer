import torch
import numpy as np
from Generator import Generator
import skimage.io as io
import os

def get_images(root_dir, num):
    x = os.listdir(root_dir)
    im_names = np.random.choice(x, size = num, replace = False)
    im_list = []
    for item in im_names:
        if item.endswith(".jpg"):
            image = io.imread(root_dir + "/" + item)
            image = np.moveaxis(image,-1,0)
            im_list.append(image)
    return torch.tensor(im_list).float(), im_names

def save_img(image, save_name):
    image = np.moveaxis(image,0,-1)
    io.imsave(save_name, image)

def load_gen(gen_path):
    device = torch.device('cpu')
    model = Generator()
    model.load_state_dict(torch.load(gen_path, map_location=device))
    model.eval()
    return model

def main(gen_path, root_dir,target_dir, num):
    images, im_names = get_images(root_dir, num)
    model = load_gen(gen_path)
    with torch.no_grad():
        output = model(images).numpy()
    for i in range(num):
        save_img(output[i], target_dir + "/" + im_names[i])

if __name__ == "__main__":
    gen_path = "/Users/gerald/Desktop/results/generator.pt"
    root_dir = "/Users/gerald/Desktop/GAN datasets/humanfaces128"
    target_dir = "/Users/gerald/Desktop/results"
    main(gen_path, root_dir, target_dir, 10)
