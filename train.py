import os
import cv2 as cv
from data_pipeline.dataloaders import get_butterfly_dataloader, \
                                        get_sift_dataloader, \
                                        get_pretrained_imagenet_dataloader
import pandas as pd
from data_pipeline.utils import read_images
from torch.utils.data import DataLoader
from train_cnn import train_baseline_net
from argparser import get_argparser

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    training_indices = pd.read_csv(args.training_index_file, sep=' ', header=None)
    training_labels = training_indices.iloc[:, 1]
    training_images = read_images(args.image_root, training_indices, 1000, gray=False)
    #sift_dataloader = get_sift_dataloader(training_images[:100], training_labels[:100], 'features/sift_features'+str(n), 1, 100)
    #butterfly_dataloader = get_butterfly_dataloader(args.image_root, args.training_index_file, args.species_file, 32)
    #train_baseline_net(butterfly_dataloader)
    imagenet_feature_dataloader = get_pretrained_imagenet_dataloader(training_images, training_labels[:1000], 32)
    for x, y in imagenet_feature_dataloader:
        import ipdb; ipdb.set_trace()
