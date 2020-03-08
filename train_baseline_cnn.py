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
    training_indices = pd.read_csv(args.training_index_file, sep=' ', header=None)
    label_i = 1
    batch_size = 64
    # normalised bgr
    bgr_training_butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.training_index_file, args.species_file, batch_size=batch_size, label_i=label_i, color_space='bgr')
    bgr_development_butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.development_index_file, args.species_file, batch_size=batch_size, label_i=label_i, color_space='bgr')
    rgb_cnn = BaselineCNN()
    run_baseline_training(rgb_cnn, 'data_pipeline/saved_models/baseline_cnn_checkpoint_bgr_normalised', bgr_training_butterfly_dataloader, bgr_development_butterfly_dataloader, resume=False, epochs=15)
    # hsv
    hsv_training_butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.training_index_file, args.species_file, batch_size=batch_size, label_i=label_i, color_space='hsv')
    hsv_development_butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.development_index_file, args.species_file, batch_size=batch_size, label_i=label_i, color_space='hsv')
    hsv_cnn = BaselineCNN()
    run_baseline_training(hsv_cnn, 'data_pipeline/saved_models/baseline_cnn_checkpoint_hsv', hsv_training_butterfly_dataloader, hsv_development_butterfly_dataloader, resume=False, epochs=15)
    # YCrCb
    hsv_training_butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.training_index_file, args.species_file, batch_size=batch_size, label_i=label_i, color_space='ycrcb')
    ycrcb_development_butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.development_index_file, args.species_file, batch_size=batch_size, label_i=label_i, color_space='ycrcb')
    ycrcb_cnn = BaselineCNN()
    run_baseline_training(ycrcb_cnn, 'data_pipeline/saved_models/baseline_cnn_checkpoint_ycrcb', ycrcb_training_butterfly_dataloader, ycrcb_development_butterfly_dataloader, resume=False, epochs=15)
