import pandas as pd
import argparse
from data_pipeline.dataloaders import get_pretrained_imagenet_dataloader,\
                                            get_sift_dataloader,\
                                                get_coloured_sift_dataloader,\
                                                    get_baseline_cnn_dataloader
from data_pipeline.utils import read_images, get_all_data_from_loader
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

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
    parser.add_argument("-sift-size", "--sift-feature-size", type=int, help="The feature size for SIFT")
    parser.add_argument("-sift-path", "--sift-feature-path", type=str, help="The path to SIFT features")
    parser.add_argument("-N", "--no-images", required=True, type=int, help="The amount of images to use in building features")
    parser.add_argument("-l", "--label-index", required=True, type=int, help="Which index to use as the label, between 1 and 5. Use 1 o classify species, 5 to classify families.")
    parser.add_argument("-ex", "--imagenet-extractor-path", type=str, help="Path to model pretrained with Imagenet and trained with transfer learning")
    parser.add_argument("-imagenet", "--imagenet-features", type=str, help="Path to imagenet features")
    parser.add_argument("-b-cnn", "--baseline-cnn-path", type=str, help="Path to trained baseline CNN")
    parser.add_argument("-cnn-feat", "--cnn-features", type=str, help="Path to baseline CNN features")
    parser.add_argument("-kernel", "--svm-kernel", default="linear", help="SVM kernel to use in classification")
    parser.add_argument("-g-cnn", "--cnn-grey", default=False, action="store_true")
    parser.add_argument("-gccnn", "--cnn-color-space", type=str, default=None, help="Color space to use in baseline CNN features")
    # TODO: add reduced dims
    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    N = args.no_images
    batch_size = N
    label_i = args.label_index
    training_indices = pd.read_csv(args.training_index_file, sep=' ', header=None)
    training_labels = training_indices.iloc[:, label_i]
    training_labels = training_labels - 1
    training_images = read_images(args.image_root, training_indices, N, grey=False)
    #imagenet_feature_dataloader = get_pretrained_imagenet_dataloader(training_images, training_labels[:N], training_labels.nunique(), \
    #                                                                    batch_size, args.imagenet_features, args.imagenet_extractor_path)
    #imagenet_features, imagenet_labels = get_all_data_from_loader(imagenet_feature_dataloader)
    baseline_cnn_feature_dataloader = get_baseline_cnn_dataloader(training_images, training_labels[:N], training_labels.nunique(), \
                                                                        batch_size, args.cnn_features, args.baseline_cnn_path, args.cnn_color_space, args.cnn_grey)
    baseline_cnn_features, baseline_cnn_labels = get_all_data_from_loader(baseline_cnn_feature_dataloader)
    #sift_dataloader = get_sift_dataloader(training_images, training_labels[:N], args.sift_feature_path, 32, feature_size=args.sift_feature_size)
    #sift_features, sift_labels = get_all_data_from_loader(sift_dataloader)
    #hsv_sift_dataloader = get_coloured_sift_dataloader(training_images, training_labels[:N], 'features/coloured_sift_hsv', 32, 'hsv', feature_size=args.sift_feature_size)
    #hsv_sift_features, hsv_sift_labels = get_all_data_from_loader(hsv_sift_dataloader)
    print('Got features')
    classifier = SVC(kernel=args.svm_kernel)
    print('Running cross validation')
    imagenet_scores = cross_val_score(classifier, baseline_cnn_features, baseline_cnn_labels, cv=3)
    print('Imagenet scores', imagenet_scores)
    #sift_scores = cross_val_score(classifier, sift_features, sift_labels, cv=3)
    #print('SIFT scores', sift_scores.mean())
    #hsv_sift_scores = cross_val_score(classifier, hsv_sift_features, labels, cv=2)
    #print('Imagenet scores', hsv_sift_scores.mean())
