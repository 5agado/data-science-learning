import numpy as np


def preprocess_images(images):
    #images = images.reshape(images.shape[0], *INPUT_SHAPE).astype('float32')
    images = (images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    images = images[:, :, :, np.newaxis]
    return images.astype('float32')
