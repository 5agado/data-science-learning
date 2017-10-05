from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers

# generic conv layer
def add_conv_layer(model, filters, kernel_size, strides, activation, input_shape=None, pool_size=None):
    if input_shape:
        model.add(Convolution2D(filters, kernel_size=kernel_size,
                             strides=strides, input_shape=input_shape))
    else:
        model.add(Convolution2D(filters, kernel_size=kernel_size, strides=strides))
    model.add(activation)
    if pool_size:
        model.add(MaxPooling2D(pool_size=pool_size))

# generic cnn model
def get_basic_model(num_classes, init_filters=32, kernel_size=(3,3), strides=(1,1),
                         input_shape=None,
                         pool_size=(2, 2), lr=0.0001, n_conv_layers=3, dense_size=64):
    #activation = Activation('relu')
    activation = LeakyReLU(0.2)


    model = Sequential()
    add_conv_layer(model, filters=init_filters, kernel_size=kernel_size,
                             strides=strides, input_shape=input_shape,
                             activation=activation, pool_size=pool_size)

    for i in range(1,n_conv_layers):
        add_conv_layer(model, filters=init_filters*(2**i), kernel_size=kernel_size,
                                 strides=strides,
                             activation=activation, pool_size=pool_size)

    model.add(Flatten())
    model.add(Dense(dense_size))
    model.add(activation)
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax')) # multiclass case
    #model.add(Activation('sigmoid')) #binary case

    optimizer = optimizers.RMSprop(lr=lr)
    #optimizer = optimizers.Adam(lr=lr)
    #model.compile(loss='binary_crossentropy',
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model

#model.fit(train_data, data_utils.label_encoder(train_df['Class'].values, encode=True),
#              nb_epoch=20, batch_size=32)