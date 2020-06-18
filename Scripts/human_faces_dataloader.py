import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
import pandas as pd
from skimage import io

class GANDataset(Dataset):
    """
    Dataset for human faces, must do the train test split before and call
    torch.tensor on its outputs, the channels are flipped already
    """
    def __init__(self,txtfilepath,root_dir, transform = None):
        self.dir = root_dir
        self.transform = transform
        self.array = pd.read_csv(txtfilepath, sep=" ", header=None)[0].values.tolist()
    def __len__(self):
        return len(self.array)

    def __getitem__(self,idx):
        img_name = os.path.join(self.dir,self.array[idx])
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        image = np.moveaxis(image,-1,0)
        return image

def get_data_loader(txtpath, root_dir, args, transform = None):
    data_set = GANDataset(txtpath,root_dir, transform)
    dataloader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
    return dataloader

#for testing only
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size',nargs="?" ,type=int, default = 2)
    args = parser.parse_args()
    y = get_data_loader("/Users/gerald/Desktop/GAN_Style_Transfer/identity_CelebA.txt", "/Users/gerald/Desktop/GAN datasets/humanfaces",args)
    for i, sample in enumerate(y):
        x = sample
        print(x.shape)
        break

