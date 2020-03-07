from argparser import get_argparser
import pandas as pd
from data_pipeline.dataloaders import get_pretrained_imagenet_dataloader
from data_pipeline.utils import read_images

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    batch_size = 32
    N = 1000
    training_indices = pd.read_csv(args.training_index_file, sep=' ', header=None)
    label_i = 1
    training_labels = training_indices.iloc[:, label_i]
    training_images = read_images(args.image_root, training_indices, N, gray=False)
    imagenet_feature_dataloader = get_pretrained_imagenet_dataloader(training_images, training_labels[:N], 200, \
                                                                        batch_size, 'features/imagenet_features_'+str(N), args.imagenet_extractor_path)
