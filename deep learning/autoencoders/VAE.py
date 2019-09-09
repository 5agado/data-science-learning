import numpy as np
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense, Flatten, GlobalMaxPooling2D, Input, Reshape
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import LeakyReLU, BatchNormalization, Lambda
from tensorflow.python.keras import backend as K
import tensorflow as tf

from autoencoder_utils import encoder_conv_block, decoder_deconv_block, get_initial_size, PlotData

# NOTES
# * some just keep 2D + maxpool and skip the flattening in the encoder


class VAE:
    def __init__(self, input_shape, config):
        self.config = config
        self.input_shape = input_shape

        self._build_model()

    def _build_model(self):
        model_input = Input(shape=self.input_shape)

        # Encoder
        self.encoder = VAE._build_encoder(model_input, self.config['model']['encoder'])

        # Decoder
        self.decoder = VAE._build_decoder(self.input_shape, self.config['model']['decoder'])

        # Full model
        mean_vector, log_var_vector, latent_vector = self.encoder(model_input)
        output = self.decoder(latent_vector)

        model = Model(inputs=[model_input], outputs=output)

        # Loss

        # Need to encapsulate all in Lambda
        # see https://github.com/tensorflow/tensorflow/issues/27112

        @tf.function()  # ??why need this (otherwise getting 'Tensor' object has no attribute '_cpu_nograd')
        def kl_loss(y_true, y_pred):
            tmp_sum = K.sum(1 + log_var_vector - K.exp(log_var_vector) - K.square(mean_vector), axis=-1)
            latent_loss = Lambda(lambda x: -0.5 * x)(tmp_sum)
            latent_loss = Lambda(lambda x: x / 784.)(K.mean(latent_loss))
            return latent_loss

        @tf.function()
        def rmse_loss(y_true, y_pred):
            loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
            return Lambda(lambda x: self.config['training']['rmse_loss_factor'] * x)(loss)  # weight reconstruction loss

        @tf.function()
        def vae_loss(y_true, y_pred):
            return kl_loss(y_true, y_pred) + rmse_loss(y_true, y_pred)

        # Compile
        optimizer = Adam(lr=self.config['training']['learning_rate'])
        model.compile(loss=vae_loss, optimizer=optimizer, metrics=[rmse_loss, kl_loss])
        #model.compile(loss=[kl_loss, rmse_loss], loss_weights=[1., 1.], optimizer=optimizer)

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
    # takes an image and generates two vectors: means and standards deviations
    def _build_encoder(model_input, config):
        latent_dim = config['latent_dim']

        x = model_input
        for i in range(config['num_conv_blocks']):
            x = encoder_conv_block(filters=config['init_filters'] * (2 ** i), block_input=x,
                                   kernel_size=config['kernel_size'], strides=config['strides'])

        features = Flatten()(x)
        features_vec = Dense(latent_dim)(features)

        # gaussian parameters
        mean_vector = Dense(latent_dim, activation='linear', name='mu')(features)
        log_var_vector = Dense(latent_dim, activation='linear', name='log_var')(features)

        encoder = Model(inputs=[model_input],
                        outputs=[mean_vector, log_var_vector, features_vec])
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
