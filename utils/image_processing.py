import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import os
import cv2

# load image from filepath and optionally resize
def load_image(filepath, img_size=None):
    #img = cv2.imread(filepath)
    img = plt.imread(filepath)
    if img_size:
        #img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
        img = resize(img, img_size, mode='reflect', preserve_range=True)
    return img

# load all images data from filepaths
def load_data(imgs_filepaths, img_size=None):
    data = np.array([load_image(img, img_size) for img in imgs_filepaths])
    return data

# load entirety of image data in batches
def load_data_batches(imgs_filepaths, img_size=None, batch_size=64):
    steps = (len(imgs_filepaths) // batch_size) + 1
    for i in range(steps):
        low = i * batch_size
        high = i * batch_size + batch_size
        print("Batch {}/{} [{}, {})".format(i, steps, low, high))
        data = load_data(imgs_filepaths[low:high], img_size)
        yield data

def image_generator(imgs_info, img_shape, num_classes, target_column, batch_size=1, img_size=None,
                    processing_pipeline=None, label_lookup=None, image_gen=None):
    """
    Image generator from Pandas dataframe. Performs data loading, options preprocessing and label lookup.

    :param imgs_info: Pandas dataframe with images info (filepath, labels)
    :param img_shape: shape of the image as returned in the data batch
    :param num_classes: number of classes for your data (used to shape the labels batch)
    :param target_column: name of column from where to get labels info
    :param batch_size: number of samples for each batch
    :param img_size: size for loaded images. No resize done if None, but should guarantee images have all same size.
    :param processing_pipeline: list of function to "chainly" apply to each data batch
    :param label_lookup: function to get proper representation of label value
    :param image_gen: image data generator to use instead of returning basic batch
    :yields tuples of (x, y) where x is a image data batch and y is the corresponding labels batch. The generator loops indefinitely.
    """

    while True:
        # setup batch data structure
        batch_data = np.zeros((batch_size, *img_shape))
        batch_labels = np.zeros((batch_size, num_classes))

        # sample entries from dataframe
        sampled_data = imgs_info.sample(min(batch_size, len(imgs_info)))

        # for each entry load image and set corresponding label
        for i, (index, row) in enumerate(sampled_data.iterrows()):
            # check if image exists
            img_path = row['filepath']
            if not os.path.isfile(img_path):
                print("{} not present".format(img_path))
            else:
                batch_data[i] = load_image(img_path, img_size).reshape(img_shape)
                # if no label lookup consider valid value in target column
                if not label_lookup:
                    batch_labels[i] = row[target_column]
                else:
                    batch_labels[i] = label_lookup(row[target_column])

        # apply functions in pipeline to data batch
        if processing_pipeline:
            for fun in processing_pipeline:
                batch_data = fun(batch_data)

        # TODO validate
        # use provided image generator to augment next batch
        if image_gen:
            gen = image_gen.flow(batch_data, batch_labels,
                                 batch_size=batch_size, shuffle=True)
            batch_data, batch_labels = next(gen)

        yield batch_data, batch_labels

def normalizeRGB(batch):
    total = np.sum(batch, axis=(1,2))
    batch /= total[:,np.newaxis,np.newaxis,:]*255
    return batch

def normalize(batch):
    # zero center
    batch -= np.mean(batch, axis=0)
    # normalize
    batch /= np.std(batch, axis=0)
    #return batch / 255
    return batch

def rotate_90(img):
    img = np.rot90(np.asarray(img))
    return Image.fromarray(np.uint8(img))

def crop(img, side=300):
    center_x = img.size[0]//2
    center_y = img.size[1]//2
    half_side = side//2
    c_img = img.crop((
            center_x - half_side,
            center_y - half_side,
            center_x + half_side,
            center_y + half_side))
    return c_img