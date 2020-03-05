import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2 as cv


class ButterflyDataset(Dataset):
    """Butterfly 200 dataset."""

    def __init__(self, indices_file, species_file, root_dir, grey, label_i, transform=None, length=None):
        """
        Args:
            indice_file (string): Path to the csv file split annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.indices = pd.read_csv(indices_file, sep=' ', header=None)
        self.idx2species = pd.read_csv(species_file, sep=' ', index_col=0, header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.grey = grey
        self.length = length if length is not None else len(self.indices)
        self.label_i = label_i

    def __len__(self):
        return self.length

    def index2species(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return [self.idx2species.loc[i].values[0] for i in index]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label_index = self.indices.iloc[idx, self.label_i]
        img_path = os.path.join(self.root_dir, self.indices.iloc[idx, 0])
        image = cv.imread(img_path)
        if self.grey:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        sample = (image, label_index)

        if self.transform:
            sample = self.transform(sample)

        return sample
