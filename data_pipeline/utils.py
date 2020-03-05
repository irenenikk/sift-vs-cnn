import cv2 as cv
import os

def read_images(root_path, indices, N=None, gray=True):
    print('Reading images from ', root_path)
    images = []
    for index in range(len(indices)):
        image_path = os.path.join(root_path, indices.iloc[index, 0])
        image = cv.imread(image_path)
        if gray:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        images.append(image)
        if N is not None and index == N-1:
            break
    print('Read', len(images), 'images')
    return images

# Rescale and ToTensor taken from this tutorial: 
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample
        img = transform.resize(image, (self.output_size, self.output_size))
        return img, label

class SampleToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        # the dataset labels are indexed from one
        label_index = label - 1
        return torch.from_numpy(image).float(), label_index

class ToTensor(object):
    """Convert ndarrays to Tensors."""

    def __call__(self, image):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return image
