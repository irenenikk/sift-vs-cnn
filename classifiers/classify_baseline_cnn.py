import sys
sys.path.append('./')

import argparse
import pandas as pd

from data_pipeline.dataloaders import get_baseline_cnn_dataloader
from data_pipeline.utils import read_images, get_all_data_from_loader
from utils import get_indices_and_labels
from training.cnn_training import evaluate_model_accuracy
from svm_classifier import classify

def get_argparser():
    parser = argparse.ArgumentParser(description='Obtain SIFT features for training set')
    parser.add_argument("-root", "--image-root", type=str, default="data/images_small",
                        help="The path to the image data folder")
    parser.add_argument("-train-idx", "--training-index-file", type=str, default="data/Butterfly200_train_release.txt",
                        help="The path to the file with training indices")
    parser.add_argument("-test-idx", "--test-index-file", type=str, default="data/Butterfly200_test_release.txt",
                        help="The path to the file with test indices")
    parser.add_argument("-s", "--species-file", type=str, default="data/species.txt",
                        help="The path to the file with mappings from index to species name")
    parser.add_argument("-N", "--no-images", required=True, type=int, help="The amount of images to use in building features")
    parser.add_argument("-l", "--label-index", required=True, type=int, help="Which index to use as the label, between 1 and 5. Use 1 o classify species, 5 to classify families.")
    parser.add_argument("-b-cnn", "--baseline-cnn-path", required=True, type=str, help="Path to trained baseline CNN")
    parser.add_argument("-cnn-feat", "--cnn-features", required=True, type=str, help="Path to baseline CNN features")
    parser.add_argument("-cnn-test-feat", "--cnn-test-features", required=True, type=str, help="Path to baseline CNN features")
    parser.add_argument("-kernel", "--svm-kernel", default="linear", help="SVM kernel to use in classification")
    parser.add_argument("-g", "--grey", default=False, action="store_true")
    parser.add_argument("-c", "--color-space", type=str, default=None, help="Color space to use in baseline CNN features")
    # TODO: add reduced dims
    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    N = args.no_images
    test_N = 1000
    label_i = args.label_index
    training_indices, training_labels = get_indices_and_labels(args.training_index_file, args.label_index)
    training_images = read_images(args.image_root, training_indices, N, grey=False)
    test_indices, test_labels = get_indices_and_labels(args.test_index_file, args.label_index)
    test_images = read_images(args.image_root, test_indices, test_N, grey=False)
    baseline_cnn_feature_dataloader = get_baseline_cnn_dataloader(training_images, training_labels[:N], training_labels.nunique(), \
                                                                        32, args.cnn_features, args.baseline_cnn_path, args.color_space, args.grey)
    test_baseline_cnn_feature_dataloader = get_baseline_cnn_dataloader(test_images, test_labels[:test_N], training_labels.nunique(), \
                                                                        32, args.cnn_test_features, args.baseline_cnn_path, args.color_space, args.grey)
    classifier = classify(baseline_cnn_feature_dataloader, test_baseline_cnn_feature_dataloader, args.svm_kernel)
