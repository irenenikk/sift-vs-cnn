import os
import cv2 as cv
from data_pipeline.dataloaders import get_butterfly_dataloader, \
                                        get_sift_dataloader, \
                                        get_coloured_sift_dataloader
import pandas as pd
from data_pipeline.utils import read_images
from torch.utils.data import DataLoader
from train_cnn import train_neural_net, find_hyperparameters
from argparser import get_argparser
from PIL import Image
from torchvision import transforms
from train_cnn import run_transfer_learning

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    training_indices = pd.read_csv(args.training_index_file, sep=' ', header=None)
    label_i = 1
    training_butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.training_index_file, args.species_file, 64, label_i)
    development_butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.development_index_file, args.species_file, 64, label_i)
    run_transfer_learning(args.model_name, training_butterfly_dataloader, development_butterfly_dataloader, training_indices.iloc[:, label_i].nunique(), resume=False, epochs=20)
