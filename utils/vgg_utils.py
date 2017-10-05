from keras.applications import vgg16
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras.layers.pooling import GlobalAveragePooling2D

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