import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import tensorflow as tf
import numpy as np
import io

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose

# TODO
# * alternatives to **kwargs, more structured config, pass also to generator and discriminator.
# * best params for Conv2DTranspose
# * add pixelshuffler
# * different kernel size for last gen layer??


# utility for the standard convolution block used in the discriminator
def conv(filters, kernel_size, strides, leaky_relu_slope=0.3, **kwargs):
    def block(x):
        x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
        x = LeakyReLU(leaky_relu_slope)(x)
        x = BatchNormalization()(x)
        return x
    return block


# utility for the standard deconvolution block used in the generator
def upscale(filters, kernel_size, strides, leaky_relu_slope=0.3, upscale_method='UPSAMPLING', **kwargs):
    def block(x):
        if upscale_method == 'UPSAMPLING':
            x = UpSampling2D()(x)
            x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
        elif upscale_method == 'DECONV':
            x = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
        else:
            raise Exception("No such upscale method: {}".format(upscale_method))
        # DCGAN paper uses LeakyReLU in the Discriminator, but pure ReLU in the Generator
        x = ReLU()(x)
        #x = LeakyReLU(leaky_relu_slope)(x)
        x = BatchNormalization()(x)
        return x
    return block


def get_generator(input_shape, init_side=7, init_filters=128, num_conv_blocks=4, n_channels=3, **kwargs):
    """
    Get generator model, which takes real values vector and generates an image
    :param input_shape:
    :param init_side:
    :param init_filters:
    :param num_conv_blocks:
    :param n_channels:
    :return:
    """
    model_input = Input(shape=input_shape)

    x = Dense(init_side * init_side * init_filters)(model_input)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Reshape((init_side, init_side, init_filters))(x)

    for i in range(1, num_conv_blocks+1):
        x = upscale(init_filters // 2 ** i, **kwargs)(x)

    upscale_method = kwargs['upscale_method']
    kernel_size = kwargs['kernel_size']
    strides = kwargs['strides']
    if upscale_method == 'UPSAMPLING':
        x = UpSampling2D()(x)
        model_output = Conv2D(n_channels, kernel_size=kernel_size, strides=strides,
                              padding='same', activation='tanh')(x)
    elif upscale_method == 'DECONV':
        model_output = Conv2DTranspose(n_channels, kernel_size=kernel_size, strides=strides,
                                       padding='same', activation='tanh')(x)
    else:
        raise Exception("No such upscale method: {}".format(upscale_method))

    return Model(model_input, model_output)


def get_discriminator(input_shape, init_filters=32, num_conv_blocks=3, **kwargs):
    """
    Get discriminator model, which takes an images and returns single real value
    :param input_shape:
    :param init_filters:
    :param num_conv_blocks:
    :return:
    """
    model_input = Input(shape=input_shape)

    x = model_input
    for i in range(num_conv_blocks):
        x = conv(init_filters * (2 ** i), **kwargs)(x)

    features = Flatten()(x)

    model_output = Dense(1, activation='linear')(features)

    return Model(inputs=[model_input], outputs=model_output)


# Add second parameter for Keras compatibility
def generator_loss(generated_output, _=None):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output),
                                                logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output),
                                                     logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss


def get_generator_train(generator, generator_loss, generator_config):
    generator_optimizer = tf.train.AdamOptimizer(generator_config['learning_rate'],
                                                 beta1=generator_config['beta1'])
    train_generator = generator_optimizer.minimize(generator_loss, var_list=generator.trainable_weights)
    return train_generator


def compile_generator(generator, generator_loss, generator_config):
    generator_optimizer = tf.train.AdamOptimizer(generator_config['learning_rate'],
                                                 beta1=generator_config['beta1'])
    generator.compile(loss=generator_loss, optimizer=generator_optimizer)


def get_discriminator_train(discriminator, discriminator_loss, discriminator_config):
    discriminator_optimizer = tf.train.AdamOptimizer(discriminator_config['learning_rate'])
    train_discriminator = discriminator_optimizer.minimize(discriminator_loss, var_list=discriminator.trainable_weights)
    return train_discriminator


def compile_discriminator(discriminator, discriminator_loss, discriminator_config):
    discriminator_optimizer = tf.train.AdamOptimizer(discriminator_config['learning_rate'])
    discriminator.compile(loss=discriminator_loss, optimizer=discriminator_optimizer)


def train_step(images, generator, discriminator,
               generator_optimizer, discriminator_optimizer,
               batch_size, noise_dim):
    # generating noise from a normal distribution
    noise = tf.random_normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator.train_on_batch(noise)

        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

    gen_train_step = generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
    disc_train_step = discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))

    return gen_loss, disc_loss, gen_train_step, disc_train_step
