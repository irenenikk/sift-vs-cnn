from torch.utils.data import Dataset
import cv2 as cv
import pickle
from os import path
import time
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pickle

class SIFTDataset(Dataset):

    def __init__(self, images, labels, feature_folder, vocabulary_size, test=False):
        self.images = images
        self.labels = labels
        assert len(self.images) == len(self.labels)
        self.test = test
        test_id = '_test' if self.test else ''
        full_feature_path = path.join(feature_folder, 'sift_features_' + str(len(images)) + '_' + str(vocabulary_size) + test_id)
        vocabulary_path = path.join(feature_folder, 'sift_vocabulary_' + str(vocabulary_size))
        if path.exists(full_feature_path):
            print('Loading SIFT features from', full_feature_path)
            self.features = pickle.load(open(full_feature_path, "rb"))
        else:
            vocabulary = None
            if self.test:
                vocabulary = pickle.load(open(vocabulary_path, 'rb'))
            else:
                vocabulary = self.get_bow_vocabulary(self.images, vocabulary_size)
                print('Saving vocabulary to', vocabulary_path)
                with open(vocabulary_path, 'wb') as f:
                    pickle.dump(vocabulary, f)
            self.features = self.get_bow_features(self.images, vocabulary)
            pickle.dump(self.features, open(full_feature_path, "wb"))
            print('Saving SIFT features to', full_feature_path)

    def get_bow_vocabulary(self, images, vocabulary_size):
        print('Building BOW vocabulary for', len(images), 'images')
        bow_kmeans_trainer = cv.BOWKMeansTrainer(vocabulary_size)
        sift = cv.xfeatures2d.SIFT_create()
        descriptors = []
        for image in images:
            keypoints, desc = sift.detectAndCompute(image, None)
            bow_kmeans_trainer.add(desc)
            descriptors += [desc]
        print('Training Kmeans with size', vocabulary_size)
        start = time.time()
        vocabulary = bow_kmeans_trainer.cluster()
        end = time.time()
        print('Training took', (end-start)/60, 'minutes')
        return vocabulary

    def get_bow_features(self, images, vocabulary):
        print('Getting BOW features')
        sift = cv.xfeatures2d.SIFT_create()
        extract = cv.xfeatures2d.SIFT_create()
        # TODO: which matcher to use?
        flann_params = dict(algorithm = 1, trees = 5)
        matcher = cv.FlannBasedMatcher(flann_params, {})
        bow_extractor = cv.BOWImgDescriptorExtractor(extract, matcher)
        bow_extractor.setVocabulary(vocabulary)
        bow_features = []
        for image in images:
            features = bow_extractor.compute(image, sift.detect(image))
            bow_features.append(features)
        return bow_features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
