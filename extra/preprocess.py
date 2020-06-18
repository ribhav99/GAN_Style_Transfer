import numpy as np
from skimage.transform import resize
import os
from skimage import io

root_dir = "/Users/gerald/Desktop/GAN datasets/cartoonset100k"
name_list = []
for root, dirs, files in os.walk(root_dir):
    for file in files:
        file_dir = os.path.join(root, file)
        if file_dir.endswith(".png"):
            image = io.imread(file_dir)
            image = resize(image, (128,128))
            io.imsave("/Users/gerald/Desktop/GAN datasets/cartoonfaces/" + file ,image)
            name_list.append(file)
with open("/Users/gerald/Desktop/GAN_Style_Transfer/data/cartoon.txt", 'w') as f:
    for item in name_list:
        f.write(item + "\n")
