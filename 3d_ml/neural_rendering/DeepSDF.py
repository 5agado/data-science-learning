import numpy as np
import logging
import sys
from typing import List
from dataclasses import dataclass

from tensorflow import keras
from keras.layers import LayerNormalization
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras import losses
from keras.optimizers import RMSprop
from keras import backend as K

import tensorflow_addons as tfa

from ds_utils.voxel_utils import get_volume_coordinates

# Based on DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation
# [Paper](https://arxiv.org/abs/1901.05103)

@dataclass
class DeepSDF:
    nb_layers: int
    latent_dim: int
    inner_activation: str
    out_activation: str
    hidden_dims: List[int]
    norm_layers: List[int]
    weight_norm: bool
    clamp_delta: float

    def __post_init__(self):
        self.model = self._get_model()

    # TODO add latent dropout
    def _get_model(self):
        """
        Create and return DeepSDF Keras Model
        :return:
        """

        # inputs
        xyz_vec = Input(shape=(3,), name='in_xyz')
        latent_vec = Input(shape=(self.latent_dim,), name='in_latent')

        x = concatenate([xyz_vec, latent_vec])

        # layers
        for layer_idx in range(self.nb_layers-1):
            # last layer use tanh activation
            if layer_idx == self.nb_layers -2:
                layer = Dense(units=1, activation='tanh')
            else:
                layer = Dense(units=self.hidden_dims[layer_idx],
                              activation=self.inner_activation)
            # add normalization to specified layers
            # either weight-norm if specified, or layer-norm as default
            if layer_idx in self.norm_layers:
                if self.weight_norm:
                    x = tfa.layers.WeightNormalization(layer)(x)
                else:
                    x = layer(x)
                    x = LayerNormalization()(x)

        # final activation
        output = x

        # build model
        deep_sdf = Model([xyz_vec, latent_vec], output)

        return deep_sdf


    def train(self, x_train, y_train, nb_epochs: int, batch_size: int, latent):
        model = self.model

        # TODO
        # add code regularization
        model.compile(
            optimizer=RMSprop(learning_rate=1e-4),
            #loss=deep_sdf_loss(delta=self.clamp_delta),
            loss='mean_absolute_error',
        )

        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=nb_epochs,
            #callbacks=[VolumeSnapshot(latent)]
        )

        return history


def deep_sdf_loss(delta=0.1):
    """
    Custom loss as defined in the paper, clamping SDF values by the controllable parameter delta.
    :param delta:
    :return:
    """
    def loss(y_true, y_pred):
        return losses.mean_absolute_error(K.clip(y_true, -delta, delta),
                                          K.clip(y_pred, -delta, delta))
    return loss


# TODO smart way to sample from network (progressive instead of uniform)
# ??Why calling self.model.predict on-epoch-start seem to stall the training (loss does not decrease)
class VolumeSnapshot(keras.callbacks.Callback):
    def __init__(self, latent):
        super(VolumeSnapshot, self).__init__()
        self.latent = latent
        self.size = 50
        self.all_xyz = get_volume_coordinates(size=self.size)
        self.volume_snapshots = []

    def on_train_end(self, logs=None):
        np.save('suzanne_train.npy', np.array(self.volume_snapshots, dtype=np.float16))

    def on_epoch_end(self, epoch, logs=None):
        pred_voxels = self.model.predict([self.all_xyz, np.tile(self.latent, (self.all_xyz.shape[0], 1))])
        voxels = pred_voxels.reshape(tuple([self.size] * 3))

        self.volume_snapshots.append(voxels)
