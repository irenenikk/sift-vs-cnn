import argparse
import os
import cv2 as cv
from data_pipeline.dataloaders import get_butterfly_dataloader, get_sift_dataloader
import pandas as pd
from data_pipeline.utils import read_images
from torch.utils.data import DataLoader
from train_cnn import train_baseline_net

parser = argparse.ArgumentParser(description='Obtain SIFT features for training set')
parser.add_argument("-root", "--image-root", type=str, default="data/images_small",
                    help="The path to the image data folder")
parser.add_argument("-train-idx", "--training_index-file", type=str, default="data/Butterfly200_train_release.txt",
                    help="The path to the file with training indices")
parser.add_argument("-s", "--species-file", type=str, default="data/species.txt",
                    help="The path to the file with mappings from index to species name")

if __name__ == '__main__':
    args = parser.parse_args()
    training_indices = pd.read_csv(args.training_index_file, sep=' ', header=None)
    training_labels = training_indices.iloc[:, 1]
    training_images = read_images(args.image_root, training_indices, 1000)
    #sift_dataloader = get_sift_dataloader(training_images[:100], training_labels[:100], 'features/sift_features'+str(n), 1, 100)
    butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.training_index_file, args.species_file, 32)
    train_baseline_net(butterfly_dataloader)
