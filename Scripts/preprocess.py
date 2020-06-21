import numpy as np
from skimage.transform import resize
import os
from skimage import io


def make_human_text_file(args):
    ### Making text files for all image names for cartoon faces
    root_dir = args.cartoon_data_root_path
    name_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_dir = os.path.join(root, file)
            if file_dir.endswith(".png"):
                # image = io.imread(file_dir)
                # image = resize(image, (128,128))
                # io.imsave("/Users/gerald/Desktop/GAN datasets/cartoonfaces/" + file ,image)
                name_list.append(file)

    with open("/content/drive/My Drive/CSC420Project/cartoon.txt", 'w') as f:
        for item in name_list:
            f.write(item + "\n")

def make_cartoon_text_file(args):
    ### Making text files for all image names for human faces
    root_dir = args.human_data_root_path
    name_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_dir = os.path.join(root, file)
            if file_dir.endswith(".jpg"):
                # image = io.imread(file_dir)
                # image = resize(image, (128,128))
                # io.imsave("/Users/gerald/Desktop/GAN datasets/cartoonfaces/" + file ,image)
                name_list.append(file)

    with open("/content/drive/My Drive/CSC420Project/human.txt", 'w') as f:
        for item in name_list:
            f.write(item + "\n")