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

class CombinedCNNDataset(Dataset):

    def __init__(self, labels, feature_folder, grey, test=False, reduced_dims=500):
        self.labels = labels
        self.test = test
        test_id = '_test' if self.test else ''
        supported_colour_spaces = ['bgr', 'hsv', 'ycrcb']
        if grey:
            supported_colour_spaces.append('grey')
        self.features = None
        for i, colour_space in enumerate(supported_colour_spaces):
            full_feature_path = path.join(feature_folder, 'baseline_cnn_features_' + colour_space + test_id)
            if path.exists(full_feature_path):
                print('Loading CNN features from', full_feature_path)
                colour_features = torch.load(full_feature_path, map_location=torch.device('cpu'))
                colour_features = [f[0].cpu().numpy() for f in colour_features]   
                assert len(colour_features) == len(labels)
                if self.features is None:
                    self.features = colour_features
                else:
                    self.features = np.concatenate((self.features, colour_features), axis=1)
            else:
                raise ValueError('Could not find path', full_feature_path)
        if reduced_dims is not None:
            reduced_features_path = full_feature_path + "_reduced_" + str(reduced_dims)
            if path.exists(reduced_features_path):
                print('Loading reduced imagened features from', reduced_features_path)
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
