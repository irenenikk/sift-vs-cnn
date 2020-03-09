import os
import argparse
import cv2 as cv
from data_pipeline.dataloaders import get_butterfly_dataloader, \
                                        get_sift_dataloader, \
                                        get_coloured_sift_dataloader
import pandas as pd
from data_pipeline.utils import read_images
from torch.utils.data import DataLoader
from cnn_training import train_neural_net, find_hyperparameters
from PIL import Image
from torchvision import transforms
from cnn_training import run_transfer_learning

def get_argparser():
    parser = argparse.ArgumentParser(description='Obtain SIFT features for training set')
    parser.add_argument("-root", "--image-root", type=str, default="data/images_small",
                        help="The path to the image data folder")
    parser.add_argument("-train-idx", "--training-index-file", type=str, default="data/Butterfly200_train_release.txt",
                        help="The path to the file with training indices")
    parser.add_argument("-dev-idx", "--development-index-file", type=str, default="data/Butterfly200_val_release.txt",
                        help="The path to the file with development indices")
    parser.add_argument("-s", "--species-file", type=str, default="data/species.txt",
                        help="The path to the file with mappings from index to species name")
    parser.add_argument("-m", "--model-name", required=True, type=str, help="Model to use in transfer learning")
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="Batch size to use in training.")
    parser.add_argument("-e", "--epochs", default=15, type=int, help="Number of training epochs.")
    parser.add_argument("-r", "--resume", default=False, action="store_true", help="If training should be resumed from model checkpoint.")
    parser.add_argument("-check", "--model-checkpoint", type=str, default="data_pipeline/saved_models/transfer_learning_checkpoint", help="Model checkpoint.")
    return parser

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    training_indices = pd.read_csv(args.training_index_file, sep=' ', header=None)
    training_butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.training_index_file, args.species_file, args.batch_size, args.label_i)
    development_butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.development_index_file, args.species_file, args.batch_size, args.label_i)
    run_transfer_learning(args.model_name, args.model_checkpoint, training_butterfly_dataloader, development_butterfly_dataloader, training_indices.iloc[:, args.label_i].nunique(), resume=args.resume, epochs=args.epochs)
