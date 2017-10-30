from keras.applications import vgg16
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras.layers.pooling import GlobalAveragePooling2D
from keras.applications.vgg16 import preprocess_input
from keras import backend as K

import numpy as np

def get_vgg(num_classes):
    vgg = vgg16.VGG16()

    vgg.layers.pop()
    for layer in vgg.layers: layer.trainable=False

    x = vgg.output
    #predictions = Dense(num_classes, activation='sigmoid')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def get_lrg_layers(n_filter, num_classes):
    return [
        #BatchNormalization(axis=1, input_shape=input_shape),
        BatchNormalization(axis=1),
        Convolution2D(n_filter, (3,3), activation='relu', padding='same'),
        BatchNormalization(axis=1),
        #MaxPooling2D(),
        Convolution2D(n_filter, (3,3), activation='relu', padding='same'),
        BatchNormalization(axis=1),
        #MaxPooling2D(),
        Convolution2D(n_filter, (3,3), activation='relu', padding='same'),
        BatchNormalization(axis=1),
        #MaxPooling2D((1,2)),
        Convolution2D(num_classes, (3,3), padding='same'),
        Dropout(0.2),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ]

def gram_matrix(x):
    # We want each row to be a channel, and the columns to be flattened x,y locations
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    # The dot product of this with its transpose shows the correlation
    # between each pair of channels
    return K.dot(features, K.transpose(features)) / x.get_shape().num_elements() #?? some didn't apply this division

# apply VGG preprocessing to batch (assumes channel last)
rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
def preprocess(x):
    #return preprocess_input(x)
    # remove zero-center by mean pixel, 'BGR'->'RGB'
    return (x - rn_mean)[:, :, :, ::-1]

# apply inverse VGG preprocessing to batch (assumes channel last)
def deprocess(x, shape):
    return np.clip(x.reshape(shape)[:, :, :, ::-1] + rn_mean, 0, 255)

#if backend.image_data_format() == 'channels_first':
#        x = x.reshape((3, height, width))
#        x = x.transpose((1, 2, 0))
#    else:
#        x = x.reshape((height, width, 3))