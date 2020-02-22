import argparse
import os
from butterfly_dataset import ButterflyDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence
from torchvision import transforms
from skimage import transform
import torch
import cv2 as cv

parser = argparse.ArgumentParser(description='Obtain SIFT features for training set')
parser.add_argument("-root", "--image_root", type=str, default="data/images_small",
                    help="The path to the image data folder")
parser.add_argument("-train-idx", "--training_index_file", type=str, default="data/Butterfly200_train_release.txt",
                    help="The path to the file with training indices")
parser.add_argument("-s", "--species_file", type=str, default="data/species.txt",
                    help="The path to the file with mappings from index to species name")

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

if __name__ == '__main__':
    args = parser.parse_args()
    image_root = args.image_root
    training_index_file = args.training_index_file
    species_file = args.species_file
    butterfly_dataset = ButterflyDataset(indices_file=training_index_file,
                                        root_dir=image_root,
                                        species_file=species_file,
                                        transform=transforms.Compose([
                                               Rescale(256)
                                        ]))
    dataloader = DataLoader(butterfly_dataset, batch_size=5, shuffle=True)
    for x, y in dataloader:
        species_name = butterfly_dataset.index2species(y)[0]
        cv.imshow(species_name, x[0].numpy())
        cv.waitKey()
