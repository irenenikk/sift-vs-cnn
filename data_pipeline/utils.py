import cv2 as cv
import os

def read_images(root_path, indices, gray=True):
    print('Reading images from ', root_path)
    images = []
    for index in range(len(indices)):
        image_path = os.path.join(root_path, indices.iloc[index, 0])
        image = cv.imread(image_path)
        if gray:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        images.append(image)
    print('Read', len(images), 'images')
    return images