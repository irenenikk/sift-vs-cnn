import torch
from torchvision import models
from torch.utils.data import Dataset
from torchvision import transforms
import json
from os import path
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from .utils import ToTensor
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PretrainedImagenet(Dataset):

    def __init__(self, images, labels, label_amount, feature_path, extractor_path, reduced_dims=None):
        self.images = images
        self.labels = labels
        self.label_amount = label_amount
        curr_dir = path.dirname(path.realpath(__file__))
        self.extractor_path = path.join(curr_dir, extractor_path)
        assert len(self.images) == len(self.labels)
        full_feature_path = path.join(curr_dir, feature_path)
        if path.exists(full_feature_path):
            print('Loading pretrained Imagenet features from', full_feature_path)
            import ipdb; ipdb.set_trace()
            self.features = torch.load(full_feature_path, map_location=device)
        else:
            self.get_features_for_images()
            print('Saving pretrained Imagenet features to', full_feature_path)
            torch.save(self.features, full_feature_path, "wb")
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
            transforms.ToPILImage(),
            transforms.Resize(256),
            ToTensor(),
            # this is obligatory when using preatrained models from pytorch
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # enable GPU
        model = self.load_transfer_learned_extractor()
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor.eval()
        feature_extractor.to(device)
        print('Getting imagenet features for', len(self.images), 'images')
        with torch.no_grad():
            self.features = [feature_extractor(preprocess(image).unsqueeze(0).to(device)) for image in tqdm(self.images)]
        print('Got features for images')

    def get_model_name(self):
        if 'resnet18' in self.extractor_path:
            return 'resnet18'
        elif 'resnet152' in self.extractor_path:
            return 'resnet152'
        raise ValueError('Please name extractor path with model architecture')

    def load_transfer_learned_extractor(self):
        model_name = self.get_model_name()
        feature_extractor = self.get_resnet_feature_extractor_for_transfer(model_name, self.label_amount)
        if not path.exists(self.extractor_path):
            ValueError('No feature extractor found trained with transfer learning. Please train the model.')
        print('Found feature extractor trained with transfer learning')
        checkpoint = torch.load(self.extractor_path, map_location=torch.device(device))
        feature_extractor.load_state_dict(checkpoint['model_state_dict'])
        feature_extractor.eval()
        return feature_extractor

    def read_imagenet_labels(self, label_path="imagenet_class_index.json"):
        full_path = path.join(curr_dir, label_path)
        labels = json.load(open(full_path, "rb"))
        idx2label = [labels[str(k)][1] for k in range(len(labels))]
        return idx2label

    @classmethod
    def get_resnet_feature_extractor_for_transfer(self, model_name, labels_amount):
        resnet = None
        if model_name == 'resnet152':
            resnet = models.resnet152(pretrained=True, progress=True)
        elif model_name == 'resnet18':
            resnet = models.resnet18(pretrained=True, progress=True)
        else:
            raise ValueError('Please give a supported model name')
        for param in resnet.parameters():
            param.requires_grad = False
        features = resnet.fc.in_features
        print('Creating a resnet with a linear layer of shape (', features, ',', labels_amount, ')')
        resnet.fc = nn.Linear(features, labels_amount)
        return resnet
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
