import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os

def plot_features(root_dir):

    images = []
    for i in range(32):
        images.append(io.imread(os.path.join(root_dir, "feature_{}.png".format(i))))
    fig, ax = plt.subplots()
    fig, axs = plt.subplots(5, 7)
    [axi.set_axis_off() for axi in axs.ravel()]
    i,j = 0, 0

    for item in images:
        axs[i,j].imshow(item, cmap= "gray")
        j += 1
        if j == 7:
            i += 1
            j = 0
        else:
            j = j % 7
    plt.savefig(os.path.join(root_dir, "all_features.png"))
