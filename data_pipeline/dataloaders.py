import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .butterfly_dataset import ButterflyDataset
from .sift_dataset import SIFTDataset
from .imagenet_pretrained import PretrainedImagenet
from .coloured_sift_dataset import ColouredSIFTDataset
from .baseline_cnn_dataset import BaselineCNNDataset
from .utils import SampleRescale, SampleToTensor

def get_butterfly_dataloader(image_root, index_file, species_file, batch_size, label_i, grey=False, length=None, color_space=None):
    butterfly_dataset = ButterflyDataset(indices_file=index_file,
                                        root_dir=image_root,
                                        species_file=species_file,
                                        grey=grey,
                                        transform=transforms.Compose([
                                               SampleRescale(256),
                                               SampleToTensor(grey)
                                        ]),
                                        length=length,
                                        label_i=label_i,
                                        color_space=color_space)
    dataloader = DataLoader(butterfly_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_sift_dataloader(images, labels, feature_folder, batch_size, feature_size):
    sift_dataset = SIFTDataset(images, labels, feature_folder, feature_size)
    dataloader = DataLoader(sift_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_coloured_sift_dataloader(images, labels, feature_folder, batch_size, colour_space, feature_size):
    sift_dataset = ColouredSIFTDataset(images, labels, feature_folder, feature_size, colour_space)
    dataloader = DataLoader(sift_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_pretrained_imagenet_dataloader(images, labels, label_amount, batch_size, feature_path, extractor_path, reduced_dims=None):
    imagenet_dataset = PretrainedImagenet(images, labels, label_amount, feature_path, extractor_path, reduced_dims)
    dataloader = DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_baseline_cnn_dataloader(images, labels, label_amount, batch_size, feature_path, extractor_path, color_space=None, grey=False, reduced_dims=None):
    imagenet_dataset = BaselineCNNDataset(images, labels, label_amount, feature_path, extractor_path, reduced_dims, color_space=color_space, grey=grey)
    dataloader = DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True)
    return dataloader
