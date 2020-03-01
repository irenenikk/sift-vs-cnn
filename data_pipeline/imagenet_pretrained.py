import torch
from torchvision import models
from torch.utils.data import Dataset
from torchvision import transforms
import json
from os import path
import torch.nn as nn
from PIL import Image
import pickle
from sklearn.decomposition import PCA

class PretrainedImagenet(Dataset):

    def __init__(self, images, labels, feature_path, reduced_dims=None, transfer_learned=False):
        self.images = images
        self.labels = labels
        self.transfer_learned = transfer_learned
        self.curr_dir = path.dirname(path.realpath(__file__))
        assert len(self.images) == len(self.labels)
        curr_dir = path.dirname(path.realpath(__file__))
        full_feature_path = path.join(curr_dir, feature_path)
        if path.exists(full_feature_path):
            print('Loading pretrained Imagenet features from', full_feature_path)
            self.features = pickle.load(open(full_feature_path, "rb"))
        else:
            self.get_features_for_images(images)
            print('Saving pretrained Imagenet features to', full_feature_path)
            pickle.dump(self.features, open(full_feature_path, "wb"))
        if reduced_dims is not None:
            reduced_features_path = path.join(curr_dir, feature_path + "_reduced_" + reduced_dims)
            if path.exists(reduced_features_path):
                self.features = pickle.load(open(reduced_features_path, "rb"))            
            else:
                pca = PCA(n_components=reduced_dims)
                self.features = pca.fit_transform(self.features)
                pickle.dump(self.features, open(reduced_features_path, "wb"))


    def get_features_for_images(self):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            # this is obligatory when using preatrained models from pytorch
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # enable GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        feature_extractor = self.load_transfer_learned_extractor() if transfer_learned else self.get_resnet_feature_extractor()
        feature_extractor.eval()
        feature_extractor.to(device)
        print('Getting imagenet features for', len(self.images), 'images')
        with torch.no_grad():
            self.features = [feature_extractor(preprocess(Image.fromarray(image)).unsqueeze(0).to(device)) for image in self.images]

    def load_transfer_learned_extractor(self):
        transferred_extractor_path = path.join(self.curr_dir, 'saved_models/transferred_extractor')
        feature_extractor = self.get_resnet_feature_extractor_for_transfer(len(self.labels))
        if not path.exists(transferred_extractor_path):
            ValueError('No feature extractor found trained with transfer learning. Please train the model.')
        print('Found feature extractor trained with transfer learning')
        feature_extractor.load_state_dict(torch.load(transferred_extractor_path))
        feature_extractor.eval()


    def read_imagenet_labels(self, label_path="imagenet_class_index.json"):
        full_path = path.join(curr_dir, label_path)
        labels = json.load(open(full_path, "rb"))
        idx2label = [labels[str(k)][1] for k in range(len(labels))]
        return idx2label

    @classmethod
    def get_resnet_feature_extractor_for_transfer(self, labels_amount):
        resnet = models.wide_resnet50_2(pretrained=True, progress=True)
        # TODO: what is the second to last layer called?
        features = resnet.fc.in_features
        resnet.fc = nn.Linear(features, labels_amount)
        return resnet
        
    def get_resnet_feature_extractor(self):
        resnet = models.wide_resnet50_2(pretrained=True, progress=True)
        # remove the last linear layer to obtain features
        feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        return feature_extractor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]