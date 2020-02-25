import torch
from torchvision.models import wide_resnet50_2
from torch.utils.data import Dataset
from torchvision import transforms
import json
from os import path
import torch.nn as nn
from PIL import Image
import pickle

class PretrainedImagenet(Dataset):

    def __init__(self, images, labels, feature_path):
        self.images = images
        self.labels = labels
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

    def get_features_for_images(self):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            # this is obligatory when using preatrained models from pytorch
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # enable GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        feature_extractor = self.get_resnet_feature_extractor()
        feature_extractor.to(device)
        print('Getting imagenet features for', len(self.images), 'images')
        with torch.no_grad():
            self.features = [feature_extractor(preprocess(Image.fromarray(image)).unsqueeze(0).to(device)) for image in self.images]

    def read_imagenet_labels(self, label_path="imagenet_class_index.json"):
        curr_dir = path.dirname(path.realpath(__file__))
        full_path = path.join(curr_dir, label_path)
        labels = json.load(open(full_path, "rb"))
        idx2label = [labels[str(k)][1] for k in range(len(labels))]
        return idx2label

    def get_resnet_feature_extractor(self):
        resnet = wide_resnet50_2(pretrained=True, progress=True)
        resnet.eval()
        # remove the last linear layer to obtain features
        feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        return feature_extractor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
