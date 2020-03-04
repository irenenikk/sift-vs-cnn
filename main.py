import os
import cv2 as cv
from data_pipeline.dataloaders import get_butterfly_dataloader, \
                                        get_sift_dataloader, \
                                        get_pretrained_imagenet_dataloader, \
                                        get_coloured_sift_dataloader
import pandas as pd
from data_pipeline.utils import read_images
from torch.utils.data import DataLoader
from train_cnn import train_neural_net, find_hyperparameters
from argparser import get_argparser
from PIL import Image
from torchvision import transforms
from train_cnn import run_transfer_learning


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
    #N = 5000
    training_indices = pd.read_csv(args.training_index_file, sep=' ', header=None)
    #training_labels = training_indices.iloc[:, 1]
    #training_images = read_images(args.image_root, training_indices, N, gray=False)
    training_butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.training_index_file, args.species_file, 64)
    development_butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.development_index_file, args.species_file, 64)
    run_transfer_learning(training_butterfly_dataloader, development_butterfly_dataloader, training_indices.iloc[:, 1].nunique(), resume=False)
    #sift_dataloader = get_sift_dataloader(training_images[:N], training_labels[:N], 'features/sift_features', 32, feature_size=500)
    #hsv_sift_dataloader = get_coloured_sift_dataloader(training_images[:N], training_labels[:N], 'features/coloured_sift_hsv', 32, 'hsv')
    #bgr_sift_dataloader = get_coloured_sift_dataloader(training_images[:N], training_labels[:N], 'features/coloured_sift_bgr', 32, 'bgr')
    #ycrcb_sift_dataloader = get_coloured_sift_dataloader(training_images[:N], training_labels[:N], 'features/coloured_sift_ycrcb', 32, 'YCrCb')
    #train_neural_net(butterfly_dataloader)
    #imagenet_feature_dataloader = get_pretrained_imagenet_dataloader(training_images, training_labels[:n], 32, 'features/imagenet_features_'+str(n))
