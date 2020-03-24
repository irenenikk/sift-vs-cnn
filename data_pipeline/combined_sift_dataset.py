from torch.utils.data import Dataset
import cv2 as cv
import pickle
import torch
from os import path
import time
import numpy as np
from .utils import change_image_colourspace
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

class CombinedSIFTDataset(Dataset):

    def __init__(self, labels, feature_folder, vocabulary_size, grey, test=False, reduced_dims=None):
        self.labels = labels
        self.test = test
        test_id = '_test' if self.test else ''
        supported_colour_spaces = ['bgr', 'hsv', 'ycrcb']
        if grey:
            supported_colour_spaces.append('grey')
        self.features = None
        for i, colour_space in enumerate(supported_colour_spaces):
            full_feature_path = path.join(feature_folder, 'sift_features_' + colour_space + '_' + str(vocabulary_size) + test_id)
            if path.exists(full_feature_path):
                print('Loading SIFT features from', full_feature_path)
                colour_features = pickle.load(open(full_feature_path, "rb"))
                assert len(colour_features[0]) == vocabulary_size
                assert len(colour_features) == len(labels)
                if self.features is None:
                    self.features = colour_features
                else:
                    self.features = np.concatenate((self.features, colour_features), axis=1)
            else:
                raise ValueError('Could not find path', full_feature_path)
        if reduced_dims is not None:
            reduced_features_path = path.join(feature_folder, "combined_sift_reduced_" + str(reduced_dims) + test_id)
            if path.exists(reduced_features_path):
                print('Loading reduced imagener features from', reduced_features_path)
                self.features = pickle.load(open(reduced_features_path, "rb"))            
            else:
                print('Building reduced features of size', reduced_dims)
                pca = PCA(n_components=reduced_dims)
                self.features = pca.fit_transform(self.features)
                print('Saving reduced imagened features to', reduced_features_path)
                pickle.dump(self.features, open(reduced_features_path, "wb"))            
        print('Got combined features of shape', self.features.shape)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.Tensor(self.features[idx]), self.labels[idx]
