from argparser import get_argparser
import pandas as pd
from data_pipeline.dataloaders import get_pretrained_imagenet_dataloader, get_sift_dataloader, get_coloured_sift_dataloader
from data_pipeline.utils import read_images, get_all_data_from_loader
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    N = 15000
    batch_size = N
    label_i = 1
    training_indices = pd.read_csv(args.training_index_file, sep=' ', header=None)
    training_labels = training_indices.iloc[:, label_i]
    training_labels = training_labels - 1
    training_images = read_images(args.image_root, training_indices, N, gray=False)
    imagenet_feature_dataloader = get_pretrained_imagenet_dataloader(training_images, training_labels[:N], 200, \
                                                                        batch_size, 'features/imagenet_features_'+str(N), args.imagenet_extractor_path)
    imagenet_features, imagenet_labels = get_all_data_from_loader(imagenet_feature_dataloader)
    sift_dataloader = get_sift_dataloader(training_images, training_labels[:N], 'features/sift_features', 32, feature_size=args.sift_feature_size)
    sift_features, sift_labels = get_all_data_from_loader(sift_dataloader)
    #hsv_sift_dataloader = get_coloured_sift_dataloader(training_images, training_labels[:N], 'features/coloured_sift_hsv', 32, 'hsv', feature_size=args.sift_feature_size)
    #hsv_sift_features, hsv_sift_labels = get_all_data_from_loader(hsv_sift_dataloader)
    classifier = SVC(kernel='linear')
    imagenet_scores = cross_val_score(classifier, imagenet_features, imagenet_labels, cv=3)
    print('Imagenet scores', imagenet_scores)
    sift_scores = cross_val_score(classifier, sift_features, sift_labels, cv=3)
    print('SIFT scores', sift_scores.mean())
    #hsv_sift_scores = cross_val_score(classifier, hsv_sift_features, labels, cv=2)
    #print('Imagenet scores', hsv_sift_scores.mean())
