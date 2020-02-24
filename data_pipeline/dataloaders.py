from torch.utils.data import DataLoader
from torchvision import transforms
from skimage import transform
from .butterfly_dataset import ButterflyDataset
from .sift_dataset import SIFTDataset
import torch

# Rescale and ToTensor taken from this tutorial: 
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample
        img = transform.resize(image, (self.output_size, self.output_size))
        return img, label

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        # the dataset labels are indexed from one
        label_index = label - 1
        return torch.from_numpy(image).float(), label_index

def get_butterfly_dataloader(image_root, index_file, species_file, batch_size, grey=False):
    butterfly_dataset = ButterflyDataset(indices_file=index_file,
                                        root_dir=image_root,
                                        species_file=species_file,
                                        grey=grey,
                                        transform=transforms.Compose([
                                               Rescale(256),
                                               ToTensor()
                                        ]))
    dataloader = DataLoader(butterfly_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_sift_dataloader(images, labels, feature_path, batch_size, feature_size=1000):
    sift_dataset = SIFTDataset(images, labels, feature_path, feature_size)
    dataloader = DataLoader(sift_dataset, batch_size=batch_size, shuffle=True)
    return dataloader