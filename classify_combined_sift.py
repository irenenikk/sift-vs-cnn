import pandas as pd
import argparse
from data_pipeline.dataloaders import get_combined_sift_dataloader
from data_pipeline.utils import read_images, get_all_data_from_loader
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from utils import get_indices_and_labels

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
    parser.add_argument("-sift-size", "--sift-feature-size", required=True, type=int, help="The feature size for SIFT")
    parser.add_argument("-f", "--feature-folder", required=True, type=str, help="The path to SIFT feature base filename")
    parser.add_argument("-N", "--no-images", required=True, type=int, help="The amount of images to use in building features")
    parser.add_argument("-l", "--label-index", required=True, type=int, help="Which index to use as the label, between 1 and 5. Use 1 o classify species, 5 to classify families.")
    parser.add_argument("-kernel", "--svm-kernel", default="linear", help="SVM kernel to use in classification")
    parser.add_argument("-g", "--grey", default=False, action="store_true")
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
    sift_dataloader = None
    test_sift_dataloader = None
    sift_dataloader = get_combined_sift_dataloader(training_labels[:N], 32, args.feature_folder, args.sift_feature_size, grey=args.grey)
    test_sift_dataloader = get_combined_sift_dataloader(test_labels[:test_N], 32, args.feature_folder, args.sift_feature_size, grey=args.grey, test=True)
    sift_features, sift_labels = get_all_data_from_loader(sift_dataloader)
    print('Got features')
    classifier = SVC(kernel=args.svm_kernel)
    '''
    cv_scores = cross_val_score(classifier, sift_features, sift_labels, cv=3)
    print('CV scores', cv_scores.mean())
    '''
    classifier.fit(sift_features, sift_labels)
    test_sift_features, test_sift_labels = get_all_data_from_loader(test_sift_dataloader)
    score = classifier.score(test_sift_features, test_sift_labels)
    print('SIFT score', score)
