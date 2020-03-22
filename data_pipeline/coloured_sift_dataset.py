from torch.utils.data import Dataset
import cv2 as cv
import pickle
import torch
from os import path
import time
import numpy as np
from .utils import change_image_colourspace
from sklearn.cluster import MiniBatchKMeans

class ColouredSIFTDataset(Dataset):

    def __init__(self, images, labels, feature_folder, vocabulary_size, color_space, test=False):
        self.images = np.asarray(images)
        if self.images.shape[-1] == 3:
            raise ValueError('Images need to have colour to form a coloured SIFT dataset')
        self.labels = labels
        assert len(self.images) == len(self.labels)
        self.grey_images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in self.images]
        self.convert_images_to_colorspace(color_space)
        self.test = test
        self.trained_kmeans_path = os.path.join(feature_folder, 'trained_kmeans' + color_space + '_' + str(vocabulary_size))
        full_feature_path = path.join(feature_folder, 'coloured_sift_' + color_space + '_' + str(vocabulary_size))
        if path.exists(full_feature_path):
            print('Loading SIFT features from', full_feature_path)
            self.features = pickle.load(open(full_feature_path, "rb"))
        else:
            self.features = self.get_coloured_bow_features(vocabulary_size)
            pickle.dump(self.features, open(full_feature_path, "wb"))
            print('Saving SIFT features to', full_feature_path)

    def convert_images_to_colorspace(self, color_space):
        self.images = [change_image_colourspace(color_space, image) for image in self.images]

    def get_coloured_descriptors(self, image, grey_image, sift):
        # the features from different image dimensions are concatenated together
        keypoints = sift.detect(grey_image)
        concat_desc = None
        for dim in range(3):
            color_dim_image = image[:, :, dim]
            keypoints, desc = sift.compute(color_dim_image, keypoints)
            if concat_desc is None:
                concat_desc = desc
            else:
                concat_desc = np.concatenate((concat_desc, desc), axis=1)
        return concat_desc

    def get_coloured_bow_features(self, vocabulary_size):
        print('Building BOW vocabulary for', len(self.images), 'images')
        sift = cv.xfeatures2d.SIFT_create()
        kmeans = MiniBatchKMeans(n_clusters=vocabulary_size, random_state=0)
        all_descriptors = None
        image_descriptors = []
        sift = cv.xfeatures2d.SIFT_create()
        print('Getting descriptors for images')
        for image, grey_image in zip(self.images, self.grey_images):
            concat_desc = self.get_coloured_descriptors(image, grey_image, sift)
            if all_descriptors is None:
                all_descriptors = concat_desc
            else:
                all_descriptors = np.concatenate((all_descriptors, concat_desc), axis=0)
            image_descriptors.append(concat_desc)
        if self.test:
            print('Loading trained Kmeans from', self.trained_kmeans_path)
            kmeans = pickle.load(open(self.trained_kmeans_path, 'rb'))
        else:
            print('Training Kmeans with size', vocabulary_size)
            start = time.time()
            kmeans = kmeans.fit(all_descriptors)
            end = time.time()
            print('Training took', (end-start)/60, 'minutes')
            print('Saving trained Kmeans to', self.trained_kmeans_path)
            with open(self.trained_kmeans_path, 'wb') as f:
                pickle.dump(kmeans, f)
        bow_features = np.zeros((len(self.images), vocabulary_size))
        for i, descriptors in enumerate(image_descriptors):
            # the features from different image dimensions are concatenated together
            clusters = kmeans.predict(descriptors)
            bow_vector = np.histogram(clusters, bins=np.arange(vocabulary_size+1), density=True)[0]
            bow_features[i] = bow_vector
        return bow_features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.Tensor(self.features[idx]), self.labels[idx]
