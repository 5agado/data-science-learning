import numpy as np
import logging
import sys
from typing import List
from dataclasses import dataclass

from tensorflow import keras
from keras.layers import Input, Dense, concatenate, Conv3D, LeakyReLU, Activation
from keras.models import Model
from keras import losses
from keras.optimizers import RMSprop


@dataclass
class AE:
    input_size: int
    latent_dim: int
    out_activation: str
    e_nb_layers: int
    e_nb_features: List[int]
    g_nb_layers: int
    g_inner_activation: str
    g_hidden_dims: List[int]

    def __post_init__(self):
        self.model = self._get_model()

    def _get_model(self):
        """
        Create and return DeepSDF Keras Model
        :return:
        """

        # inputs
        vox_3d = Input(shape=[self.input_size]*3+[1], name='in_xyz')
        xyz_vec = Input(shape=(3,), name='in_xyz')

        self.encoder = self._build_encoder(vox_3d)
        self.generator = self._build_generator(self.encoder.output)

        # build model
        ae = Model(vox_3d, self.generator.output)

        return ae

    def _build_encoder(self, model_input):
        x = model_input
        for i in range(self.e_nb_layers):
            x = encoder_conv_3d_block(x, self.e_nb_features[i], 3, [1, 2, 2, 2, 1])

        features = encoder_conv_3d_block(x, self.e_nb_features[self.e_nb_layers-1], 3, [1, 1, 1, 1, 1], activation='sigmoid')

        encoder = Model(inputs=[model_input], outputs=features)
        return encoder

    def _build_generator(self, model_input):
        # inputs
        xyz_vec = Input(shape=(3,), name='in_xyz')
        latent_vec = Input(shape=(self.latent_dim,), name='in_latent')

        x = concatenate([xyz_vec, latent_vec])

        # layers
        for layer_idx in range(self.g_nb_layers - 1):
            # last layer use tanh activation
            if layer_idx == self.g_nb_layers - 2:
                layer = Dense(units=1, activation='tanh')
            else:
                layer = Dense(units=self.g_hidden_dims[layer_idx],
                              activation=self.g_inner_activation)
            x = layer(x)

        # final activation
        output = x

        # build model
        generator = Model([xyz_vec, latent_vec], output)
        return generator

    def train(self, x_train, y_train, nb_epochs: int, batch_size: int, latent):
        pass


def encoder_conv_3d_block(block_input, filters, kernel_size, strides, activation='lrelu'):
    block = Conv3D(filters, kernel_size, strides=strides)(block_input)
    if activation == 'lrelu':
        block = LeakyReLU()(block)
    elif activation == 'sigmoid':
        block = Activation('sigmoid')
    #block = BatchNormalization()(block)
    return block