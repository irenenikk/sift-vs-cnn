from torch.utils.data import Dataset
import cv2 as cv
import pickle
from os import path
import time
from sklearn.cluster import MiniBatchKMeans
import numpy as np

class ColouredSIFTDataset(Dataset):

    def __init__(self, images, labels, feature_path, vocabulary_size, color_space):
        self.images = np.asarray(images)
        if self.images.shape[-1] == 3:
            raise ValueError('Images need to have colour to form a coloured SIFT dataset')
        self.labels = labels
        assert len(self.images) == len(self.labels)
        self.change_image_colourspace(color_space)
        curr_dir = path.dirname(path.realpath(__file__))
        full_feature_path = path.join(curr_dir, feature_path)
        if path.exists(full_feature_path):
            print('Loading SIFT features from', full_feature_path)
            self.features = pickle.load(open(full_feature_path, "rb"))
        else:
            vocabulary = self.get_coloured_bow_vocabulary(vocabulary_size)
            self.features = self.get_coloured_bow_features(vocabulary)
            pickle.dump(self.features, open(full_feature_path, "wb"))
            print('Saving SIFT features to', full_feature_path)

    def change_image_colourspace(self, color_space):
        # opencv color order is (blue, green, red)
        transform = None
        if color_space == 'hsv':
            transform = lambda image: cv.cvtColor(image, cv.COLOR_BGR2HSV)
        elif color_space == 'YCrCb':
            transform = lambda image: cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        elif color_space == 'bgr':
            transform = lambda image: cv.normalize(image, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
        elif color_space == 'obgr':
            # TODO: implement
            pass
        else:
            raise ValueError('Color space', color_space, 'not supported')
        self.images = [transform(image) for image in self.images]

    def get_coloured_descriptors(self, image, sift):
        # the features from different image dimensions are concatenated together
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        keypoints = sift.detect(gray)
        concat_desc = None
        for dim in range(3):
            color_dim_image = image[:, :, dim]
            keypoints, desc = sift.compute(color_dim_image, keypoints)
            if concat_desc is None:
                concat_desc = desc
            else:
                concat_desc = np.concatenate((concat_desc, desc), axis=1)
        return concat_desc

    def get_coloured_bow_vocabulary(self, vocabulary_size):
        print('Building BOW vocabulary for', len(self.images), 'images')
        bow_kmeans_trainer = cv.BOWKMeansTrainer(vocabulary_size)
        sift = cv.xfeatures2d.SIFT_create()
        for image in self.images:
            # the features from different image dimensions are concatenated together
            concat_desc = self.get_coloured_descriptors(image, sift)
            bow_kmeans_trainer.add(concat_desc)
        print('Training Kmeans with size', vocabulary_size)
        start = time.time()
        vocabulary = bow_kmeans_trainer.cluster()
        end = time.time()
        print('Training took', (end-start)/60, 'minutes')
        import ipdb; ipdb.set_trace()
        # check what the the vocabulary is
        return vocabulary

    def get_coloured_bow_features(self, vocabulary):
        print('Getting BOW features')
        sift = cv.xfeatures2d.SIFT_create()
        extract = cv.xfeatures2d.SIFT_create()
        # TODO: which matcher to use?
        flann_params = dict(algorithm = 1, trees = 5)
        matcher = cv.FlannBasedMatcher(flann_params, {})
        bow_extractor = cv.BOWImgDescriptorExtractor(extract, matcher)
        bow_extractor.setVocabulary(vocabulary)
        bow_features = []
        for image in self.images:
            concat_features = self.get_coloured_descriptors(image, sift)
            bow_features.append(concat_features)
        return bow_features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
