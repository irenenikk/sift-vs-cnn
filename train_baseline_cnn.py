import os
import cv2 as cv
from data_pipeline.dataloaders import get_butterfly_dataloader
import pandas as pd
from models.baseline_cnn import BaselineCNN
from cnn_training import run_baseline_training, find_hyperparameters
from argparser import get_argparser
from PIL import Image
from torchvision import transforms

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
    N = 1000
    training_indices = pd.read_csv(args.training_index_file, sep=' ', header=None)
    label_i = 1
    batch_size = 64
    training_butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.training_index_file, args.species_file, batch_size, label_i)
    development_butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.development_index_file, args.species_file, batch_size, label_i)
    baseline_cnn = BaselineCNN(batch_size=batch_size)
    run_baseline_training(baseline_cnn, training_butterfly_dataloader, development_butterfly_dataloader, resume=False, epochs=50)
