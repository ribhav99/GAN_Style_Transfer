import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from skimage import io
from skimage.color import rgb2gray

class GANDataset(Dataset):
    """
    Dataset for human faces and cartoon faces, the channels are flipped already.

    Human faces data from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    Cartoon data from: https://google.github.io/cartoonset/
    """

    def __init__(self, args):
        self.human_dir = args.human_root_dir
        self.cartoon_dir = args.cartoon_root_dir
        self.human_array = self._get_file_list(args.human_root_dir)
        self.cartoon_array = self._get_file_list(args.cartoon_root_dir)
        self.gray = args.gray

    def _get_file_list(self, root_dir):
        x = []
        for root, dirs, files in os.walk(root_dir, topdown=False):
            for name in files:
                if name.endswith(".jpg") or name.endswith(".png"):
                    x.append(os.path.join(root, name))
        return x

    def __len__(self):
        return len(self.human_array)

    def load(self, root_dir, idstr):
        img_name = os.path.join(root_dir, idstr)
        image = io.imread(img_name).astype("float32")
        if self.gray:
            image = rgb2gray(image)
        image = (image / 127.5) - 1
        if len(image.shape) == 3:
            image = np.moveaxis(image, -1, 0)
        elif len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        return torch.from_numpy(image).float()

    def __getitem__(self, idx):
        human_face = self.load(self.human_dir, self.human_array[idx])
        cartoon_face = self.load(self.cartoon_dir, self.cartoon_array[idx])
        return human_face, cartoon_face


def get_data_loader(args):
    data_set = GANDataset(args)
    dataloader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
    return dataloader
