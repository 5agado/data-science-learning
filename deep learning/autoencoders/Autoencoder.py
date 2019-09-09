import numpy as np
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense, Flatten, GlobalMaxPooling2D, Input, Reshape
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import LeakyReLU, BatchNormalization

import tensorflow as tf

from autoencoder_utils import encoder_conv_block, decoder_deconv_block, get_initial_size, PlotData

# NOTES
# * some just keep 2D + maxpool and skip the flattening in the encoder


class Autoencoder:
    def __init__(self, input_shape, config):
        self.config = config
        self.input_shape = input_shape

        self._build_model()

    def _build_model(self):
        model_input = Input(shape=self.input_shape)

        # Encoder
        self.encoder = Autoencoder._build_encoder(model_input, self.config['model']['encoder'])

        # Decoder
        self.decoder = Autoencoder._build_decoder(self.input_shape, self.config['model']['decoder'])

        # Full model
        features = self.encoder(model_input)
        output = self.decoder(features)

        model = Model(inputs=[model_input], outputs=output)
        # problem of coupled training setup with model building
        # for example messy use of config for both model and training

        optimizer = Adam(lr=self.config['training']['learning_rate'],
                         decay=self.config['training']['beta1'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy')

        self.model = model

    def train(self, train_ds, validation_ds, nb_epochs: int, log_dir, checkpoint_dir):
        callbacks = []

        # tensorboard
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)

        # checkpoints
        if checkpoint_dir:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                             save_weights_only=True,
                                                             verbose=1,
                                                             period=self.config['training']['checkpoint_steps'])
            callbacks.append(cp_callback)

        # plotting callback
        plot_callback = PlotData(validation_ds, self.model, log_dir)
        callbacks.append(plot_callback)

        # model.fit(x_train, y_train, epochs=nb_epochs)  # "old" way, passing pure numpy data
        # when passing an infinitely repeating dataset, must specify the `steps_per_epoch` argument.
        self.model.fit(train_ds, train_ds, epochs=nb_epochs,
                       batch_size=self.config['training']['batch_size'],
                       validation_data=[validation_ds, validation_ds],
                       callbacks=callbacks)

    @staticmethod
    def _build_encoder(model_input, config):
        x = model_input
        for i in range(config['num_conv_blocks']):
            x = encoder_conv_block(filters=config['init_filters'] * (2 ** i), block_input=x,
                                   kernel_size=config['kernel_size'], strides=config['strides'])

        features = Flatten()(x)
        features = Dense(config['latent_dim'])(features)

        encoder = Model(inputs=[model_input], outputs=features)
        return encoder

    @staticmethod
    def _build_decoder(img_shape, config):
        latent_vector = Input(config['latent_dim'])
        init_shape = tuple([get_initial_size(d, config['num_conv_blocks'])
                            for d in img_shape[:-1]] + [config['init_filters']])

        # CNN part
        #x = Dense(1024)(latent_vector)
        #x = LeakyReLU()(x)

        x = Dense(np.prod(init_shape))(latent_vector)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Reshape(init_shape)(x)

        for i in range(config['num_conv_blocks']):
            x = decoder_deconv_block(filters=config['init_filters'] // (2 ** i), block_input=x,
                                     kernel_size=config['kernel_size'], strides=config['strides'])

        x = Conv2D(img_shape[-1], (2, 2), padding='same', activation='sigmoid')(x)

        # ??why not passing as input the actual features tensor, output of the encoder??
        return Model(inputs=latent_vector, outputs=x)
