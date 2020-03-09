import torch
import pickle
from torchvision import models
from torch.utils.data import Dataset
from torchvision import transforms
import json
from os import path
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from .utils import ToTensor, change_image_colourspace, Flatten, Rescale
from tqdm import tqdm
from models.baseline_cnn import BaselineCNN
import cv2 as cv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaselineCNNDataset(Dataset):

    def __init__(self, images, labels, label_amount, feature_path, extractor_path, reduced_dims=None, color_space=None, grey=False):
        self.images = images
        self.labels = labels
        self.label_amount = label_amount
        self.color_space = color_space
        self.grey = grey
        curr_dir = path.dirname(path.realpath(__file__))
        self.extractor_path = path.join(curr_dir, extractor_path)
        assert len(self.images) == len(self.labels)
        full_feature_path = path.join(curr_dir, feature_path)
        if path.exists(full_feature_path):
            print('Loading pretrained baseline CNN features from', full_feature_path)
            self.features = torch.load(full_feature_path, map_location=device)
        else:
            if not grey and color_space is not None:
                self.images = [change_image_colourspace(self.color_space, image) for image in self.images]
            elif grey:
                self.images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in self.images]
            self.get_features_for_images()
            print('Saving baseline CNN features to', full_feature_path)
            torch.save(self.features, full_feature_path)
        if reduced_dims is not None:
            reduced_features_path = path.join(curr_dir, feature_path + "_reduced_" + reduced_dims)
            if path.exists(reduced_features_path):
                print('Loading reduced imagened features from', reduced_features_path)
                self.features = pickle.load(open(reduced_features_path, "rb"))            
            else:
                print('Building reduced features of size', reduced_dims)
                pca = PCA(n_components=reduced_dims)
                self.features = pca.fit_transform(self.features)
                print('Saving reduced imagened features to', reduced_features_path)
                pickle.dump(self.features, open(reduced_features_path, "wb"))


    def get_features_for_images(self):
        preprocess = transforms.Compose([
            Rescale(256),
            ToTensor(self.grey),
        ])
        # enable GPU
        model = self.load_trained_extractor()
        print(model)
        children = list(model.children())
        feature_extractor = nn.Sequential(*list(children[:2] + [nn.ReLU()] + children[2:4] + [nn.ReLU()] + [children[4]]+ [Flatten(model.flattened_size)] + children[5:-2]))
        print(feature_extractor)
        feature_extractor.to(device)
        print('Getting Baseline CNN features for', len(self.images), 'images')
        with torch.no_grad():
            self.features = [feature_extractor(preprocess(image).unsqueeze(0).to(device)) for image in tqdm(self.images)]
        print('Got features for images')

    def load_trained_extractor(self):
        feature_extractor = BaselineCNN(grey=self.grey)
        if not path.exists(self.extractor_path):
            ValueError('No feature extractor found. Please train the baseline CNN model.')
        print('Found a baseline CNN feature extractor')
        checkpoint = torch.load(self.extractor_path, map_location=device)
        feature_extractor.load_state_dict(checkpoint['model_state_dict'])
        feature_extractor.eval()
        return feature_extractor
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
