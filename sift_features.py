import argparse
import os
import cv2 as cv
from data_pipeline.dataloader import get_butterfly_dataloader

parser = argparse.ArgumentParser(description='Obtain SIFT features for training set')
parser.add_argument("-root", "--image_root", type=str, default="data/images_small",
                    help="The path to the image data folder")
parser.add_argument("-train-idx", "--training_index_file", type=str, default="data/Butterfly200_train_release.txt",
                    help="The path to the file with training indices")
parser.add_argument("-s", "--species_file", type=str, default="data/species.txt",
                    help="The path to the file with mappings from index to species name")

if __name__ == '__main__':
    args = parser.parse_args()
    dataloader = get_butterfly_dataloader(args.image_root, args.training_index_file, args.species_file, 128)
    for x, y in dataloader: 
        species_names = butterfly_dataset.index2species(y)
        cv.imshow(species_name[0], x[0].numpy())
        cv.waitKey()
