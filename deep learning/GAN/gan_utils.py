import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, UpSampling2D, Conv2DTranspose
from tensorflow.python.keras.layers import LeakyReLU, BatchNormalization, Lambda, Activation


# TODO overlap between GAN and Autoencoders utils

# utility for the standard convolution block used in the discriminator
def conv(filters, kernel_size, strides, leaky_relu_slope=0.3, padding='same'):
    def block(x):
        x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x = LeakyReLU(leaky_relu_slope)(x)
        x = BatchNormalization()(x)
        return x
    return block


# utility for the standard deconvolution block used in the generator
def upscale(filters, kernel_size, strides, leaky_relu_slope=0.3, upscale_method='UPSAMPLING',
            activation='ReLU'):
    def block(x):
        if upscale_method == 'UPSAMPLING':
            x = UpSampling2D()(x)
            x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
        elif upscale_method == 'DECONV':
            x = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
        else:
            raise Exception("No such upscale method: {}".format(upscale_method))

        # DCGAN paper uses LeakyReLU in the Discriminator, but pure ReLU in the Generator
        if activation.lower() == 'leakyrelu':
            #x = ReLU()(x)
            x = LeakyReLU(leaky_relu_slope)(x)
        else:
            x = Activation(activation)(x)
        x = BatchNormalization()(x)
        return x
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


class PlotData(tf.keras.callbacks.Callback):
    def __init__(self, test_data, model, log_dir, sample_size=1):
        super(PlotData, self).__init__()
        self.summary_writer = tf.summary.create_file_writer(str(log_dir / 'plot'))
        self.test_data = test_data
        self.model = model
        self.sample_size = sample_size

    def on_epoch_begin(self, epoch, logs=None):
        #rand_idxs = np.random.randint(0, len(self.test_data), self.sample_size)
        #x, _ = self.test_data.take(1)
        for x, y in self.test_data:  # todo decent way to get random entry from prefetch dataset
            rand_idx = np.random.randint(len(x))
            x = x[rand_idx:rand_idx+1]
            with self.summary_writer.as_default():
                # Plot sample data
                predictions = self.model.predict(x)
                tf.summary.image("Sample Generated", predictions, step=epoch)
                tf.summary.image("Sample Input", x, step=epoch)
