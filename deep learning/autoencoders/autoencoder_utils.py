import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import backend as K


# standard convolution block used in the encoder
# TODO
# * activations options
# * optional maxpool
def encoder_conv_block(filters, block_input, kernel_size=3, strides=1, padding='same'):
    block = Conv2D(filters, kernel_size, strides=strides, padding=padding)(block_input)
    block = LeakyReLU()(block)
    block = BatchNormalization()(block)
    #block = MaxPool2D(pool_size=2)(block)
    return block


# utility for the standard deconvolution block used in the decoder
def decoder_deconv_block(filters, block_input, kernel_size=3, strides=1):
    block = UpSampling2D()(block_input)
    block = Convolution2D(filters, kernel_size, strides=strides, padding='same')(block)
    block = BatchNormalization()(block)
    block = LeakyReLU()(block)
    return block


# utility to get initial shape to get to final specified one
# based on the wanted number of deconv block.
def get_initial_size(final_size, num_deconv_blocks, factor=2):
    if num_deconv_blocks==0:
        return final_size
    else:
        return get_initial_size(final_size//factor,
                                num_deconv_blocks-1,
                                factor=factor)


# sample latent vector using learned distribution parameters
def sampling(z_mean, z_log_sigma, batch_size, latent_dim):
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_sigma / 2) * epsilon  # equivalent to e * std + mean


# Custom Keras layer: sample latent vector using learned distribution parameters
# could also wrap a Python function into a Lambda layer
class Sampling(Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = K.random_normal(tf.shape(log_var), mean=0., stddev=1.)
        sample = epsilon * K.exp(log_var / 2) + mean  # equivalent to e * std + mean
        return sample


class PlotData(tf.keras.callbacks.Callback):
    def __init__(self, test_data, model, log_dir, sample_size=1):
        super(PlotData, self).__init__()
        self.summary_writer = tf.summary.create_file_writer(str(log_dir / 'plot'))
        self.test_data = test_data
        self.model = model
        self.sample_size = sample_size

    def on_epoch_begin(self, epoch, logs=None):
        for x, y in self.test_data:  # todo decent way to get random entry from prefetch dataset
            rand_idx = np.random.randint(len(x))
            x = x[rand_idx:rand_idx+1]
            with self.summary_writer.as_default():
                # Plot sample data
                predictions = self.model.predict(x)
                tf.summary.image("Sample Generated", predictions, step=epoch)
                tf.summary.image("Sample Input", x, step=epoch)
