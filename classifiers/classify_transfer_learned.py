import sys
sys.path.append('./')

import argparse
import pandas as pd
import cv2 as cv
import numpy as np

from data_pipeline.dataloaders import get_pretrained_imagenet_dataloader
from data_pipeline.utils import read_images
from classifiers.utils import get_indices_and_labels
from classifiers.svm_classifier import classify

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
    parser.add_argument("-sift-size", "--sift-feature-size", type=int, help="The feature size for SIFT")
    parser.add_argument("-sift-path", "--sift-feature-path", type=str, help="The path to SIFT features")
    parser.add_argument("-N", "--no-images", required=True, type=int, help="The amount of images to use in building features")
    parser.add_argument("-l", "--label-index", required=True, type=int, help="Which index to use as the label, between 1 and 5. Use 1 o classify species, 5 to classify families.")
    parser.add_argument("-ex", "--imagenet-extractor-path", required=True, type=str, help="Path to model pretrained with Imagenet and trained with transfer learning")
    parser.add_argument("-imagenet", "--imagenet-features", required=True, type=str, help="Path to imagenet features")
    parser.add_argument("-kernel", "--svm-kernel", default="linear", help="SVM kernel to use in classification")
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
    imagenet_feature_dataloader = get_pretrained_imagenet_dataloader(training_images, training_labels[:N], training_labels.nunique(), \
                                                                        32, args.imagenet_features, args.imagenet_extractor_path)
    test_imagenet_feature_dataloader = get_pretrained_imagenet_dataloader(test_images, test_labels[:test_N], test_labels[:test_N].nunique(), \
                                                                        32, args.imagenet_features + '_test', args.imagenet_extractor_path)
    classifier = classify(imagenet_feature_dataloader, test_imagenet_feature_dataloader, args.svm_kernel)
    # show false predictions
    preds = classifier.predict(test_imagenet_features)
    false_pred = preds != test_imagenet_labels
    false_pred_images = np.asarray(test_images)[false_pred]
    for i in range(5):
        cv.imshow('', false_pred_images[i])
        cv.waitKey(0)

