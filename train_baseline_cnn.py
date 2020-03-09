import os
import argparse
import cv2 as cv
from data_pipeline.dataloaders import get_butterfly_dataloader
import pandas as pd
from models.baseline_cnn import BaselineCNN
from cnn_training import run_baseline_training, find_hyperparameters
from PIL import Image
from torchvision import transforms

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
    parser.add_argument("-l", "--label-index", required=True, type=int, help="Which index to use as the label, between 1 and 5. Use 1 o classify species, 5 to classify families.")
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="Batch size to use in training.")
    parser.add_argument("-e", "--epochs", default=15, type=int, help="Number of training epochs.")
    parser.add_argument("-r", "--resume", default=False, action="store_true", help="If training should be resumed from model checkpoint.")
    parser.add_argument("-check", "--model-checkpoint", type=str, default="data_pipeline/saved_models/baseline_cnn_checkpoint", help="Model checkpoint.")
    parser.add_argument("-color", "--color-space", type=str, default=None, help="Color space to use.")
    parser.add_argument("-g", "--grey", default=False, action="store_true", help="Use grey images in training.")
    return parser

def find_baseline_hyperparameters(training_images, training_labels):
    new_size = 256
    preprocess = transforms.Compose([
                transforms.Resize((new_size, new_size)),
                transforms.ToTensor()])
    resized_images = [preprocess(Image.fromarray(image)) for image in training_images]
    find_hyperparameters(resized_images, training_labels[:len(training_images)].values)


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    training_indices = pd.read_csv(args.training_index_file, sep=' ', header=None)
    training_butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.training_index_file, args.species_file, batch_size=args.batch_size, label_i=args.label_i, color_space=args.color_space)
    development_butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.development_index_file, args.species_file, batch_size=args.batch_size, label_i=args.label_i, color_space=args.color_space)
    baseline_cnn = BaselineCNN()
    run_baseline_training(baseline_cnn, args.model_checkpoint, training_butterfly_dataloader, development_butterfly_dataloader, resume=args.resume, epochs=args.epochs)
