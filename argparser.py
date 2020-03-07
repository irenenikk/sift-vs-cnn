import argparse

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
    parser.add_argument("-sift-f", "--sift-feature-size", type=int, help="Define the feature size for SIFT")
    parser.add_argument("-ex", "--imagenet-extractor-path", type=str, help="Path to model pretrained with Imagenet and trained with transfer learning")
    parser.add_argument("-m", "--model-name", type=str, help="Model to use in transfer learning")
    return parser
