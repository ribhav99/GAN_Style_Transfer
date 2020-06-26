import os
import pandas as pd
import numpy as np
import skimage.io as io
from tqdm import tqdm
from skimage import img_as_float

cartoon_train = pd.read_csv("/Users/gerald/Desktop/GAN datasets/cartoon_train.txt", sep=" ", header=None)[0].values.tolist()
human_train = pd.read_csv("/Users/gerald/Desktop/GAN datasets/human_train.txt", sep=" ", header=None)[0].values.tolist()

cartoon_root = "/Users/gerald/Desktop/GAN datasets/cartoonfacesgray"
human_root = "/Users/gerald/Desktop/GAN datasets/humangray128"

def compute_mean(train_array,root_dir, mean_save_name):
    mean_matrix = np.zeros((128,128))
    count = 0
    for root, dirs, files in os.walk(root_dir, topdown=False):
       for name in files:
            if name in train_array:
                file_path = os.path.join(root, name)
                image = io.imread(file_path)
                image = img_as_float(image)
                mean_matrix = mean_matrix + image
                count += 1
    mean_matrix = mean_matrix / count
    np.save(mean_save_name, mean_matrix)

def compute_std(train_array, root_dir, mean, std_save_name):
    std_matrix = np.zeros((128,128))
    count = 0
    for root, dirs, files in os.walk(root_dir, topdown=False):
       for name in files:
            if name in train_array:
                file_path = os.path.join(root, name)
                image = io.imread(file_path)
                image = img_as_float(image)
                curr_compute = (image - mean)**2
                std_matrix = curr_compute + std_matrix
                count += 1
    std_matrix = np.sqrt(std_matrix) / (count - 1)
    np.save(std_save_name, std_matrix)

if __name__ == "__main__":
    compute_mean(cartoon_train,cartoon_root, "cartoon_mean")
    compute_mean(human_train,human_root, "human_mean")
    human_mean = np.load("human_mean.npy")
    cartoon_mean = np.load("cartoon_mean.npy")
    compute_std(cartoon_train, cartoon_root, cartoon_mean, "cartoon_std")
    compute_std(human_train, human_root, human_mean, "human_std")
