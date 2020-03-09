import os
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
    parser.add_argument("-sift-f", "--sift-feature-size", required=True, type=int, help="The feature size for SIFT")
    parser.add_argument("-N", "--no-images", required=True, type=int, help="The amount of images to use in building features")
    parser.add_argument("-l", "--label-index", required=True, type=int, help="Which index to use as the label, between 1 and 5. Use 1 o classify species, 5 to classify families.")
    return parser

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    training_indices = pd.read_csv(args.training_index_file, sep=' ', header=None)
    N = args.no_images if N < len(training_indices) else len(training_indices)
    label_i = 1
    training_labels = training_indices.iloc[:, label_i]
    training_images = read_images(args.image_root, training_indices, N, gray=False)
    sift_dataloader = get_sift_dataloader(training_images[:N], training_labels[:N], 'features/sift_features', 32, feature_size=args.sift_feature_size)
    hsv_sift_dataloader = get_coloured_sift_dataloader(training_images[:N], training_labels[:N], 'features/coloured_sift_hsv', 32, 'hsv', feature_size=args.sift_feature_size)
    ycrcb_sift_dataloader = get_coloured_sift_dataloader(training_images[:N], training_labels[:N], 'features/coloured_sift_ycrcb', 32, 'YCrCb', feature_size=args.sift_feature_size)
    bgr_sift_dataloader = get_coloured_sift_dataloader(training_images[:N], training_labels[:N], 'features/coloured_sift_bgr', 32, 'bgr', feature_size=args.sift_feature_size)
