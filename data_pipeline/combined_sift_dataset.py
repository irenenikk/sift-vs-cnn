from torch.utils.data import Dataset
import cv2 as cv
import pickle
import torch
from os import path
import time
import numpy as np
from .utils import change_image_colourspace
from sklearn.cluster import MiniBatchKMeans

class CombinedSIFTDataset(Dataset):

    def __init__(self, labels, feature_folder, vocabulary_size, grey, test=False):
        self.labels = labels
        self.test = test
        test_id = '_test' if self.test else ''
        supported_colour_spaces = ['bgr', 'hsv', 'ycrcb']
        if grey:
            supported_colour_spaces.append('grey')
        self.features = None
        for i, colour_space in enumerate(supported_colour_spaces):
            full_feature_path = path.join(feature_folder, 'sift_features_' + color_space + '_' + str(vocabulary_size) + test_id)
            if path.exists(full_feature_path):
                print('Loading SIFT features from', full_feature_path)
                colour_features = pickle.load(open(full_feature_path, "rb"))
                assert colour_features.shape[1] == vocabulary_size
                assert colour_features.shape[0] == len(labels)
                if self.features is None:
                    self.features = colour_features
                else:
                    self.features = np.concatenate((self.features, colour_features), axis=1)
            else:
                raise ValueError('Could not find path', full_feature_path)
        print('Got combined features of shape', self.features.shape)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.Tensor(self.features[idx]), self.labels[idx]
