import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from skimage import io

class GANDataset(Dataset):
    """
    Dataset for human faces and cartoon faces, the channels are flipped already.

    Human faces data from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    Cartoon data from: https://google.github.io/cartoonset/
    """

    def __init__(self, human_file_path, cartoon_file_path, human_root_dir, cartoon_root_dir, transform=None):
        self.human_dir = human_root_dir
        self.cartoon_dir = cartoon_root_dir
        self.transform = transform
        self.human_array = pd.read_csv(human_file_path, sep=" ", header=None)[
            0].values.tolist()
        self.cartoon_array = pd.read_csv(cartoon_file_path, sep=" ", header=None)[
            0].values.tolist()
        assert len(self.human_array) == len(self.cartoon_array)

    def __len__(self):
        return len(self.human_array)

    def load(self, root_dir, idstr):
        img_name = os.path.join(root_dir, idstr)
        image = io.imread(img_name).astype("float32")
        if self.transform:
            image = self.transform(image)
        image = (image / 127.5) - 1
        image = np.moveaxis(image, -1, 0)
        return torch.from_numpy(image).float()

    def __getitem__(self, idx):
        human_face = self.load(self.human_dir, self.human_array[idx])
        cartoon_face = self.load(self.cartoon_dir, self.cartoon_array[idx])
        return human_face, cartoon_face


def get_data_loader(args, train=True, transform=None):
    human_txt_path = args.human_train_path if train is True else args.human_test_path
    cartoon_txt_path = args.cartoon_train_path if train is True else args.cartoon_test_path
    data_set = GANDataset(human_txt_path, cartoon_txt_path,
                          args.human_data_root_path, args.cartoon_data_root_path, transform)
    dataloader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
    return dataloader
