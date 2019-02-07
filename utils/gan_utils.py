from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.optimizers import Adam

import plot_utils


# utility for the standard convolution block used in the discriminator
def conv(filters, kernel_size=5, strides=2):
    def block(x):
        x = Conv2D(filters, kernel_size=kernel_size,
                   strides=strides, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = BatchNormalization()(x)
        return x
    return block


# utility for the standard deconvolution block used in the generator
def upscale(filters, kernel_size=3):
    def block(x):
        x = UpSampling2D()(x)
        x = Conv2D(filters, kernel_size=kernel_size,
                   padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = BatchNormalization()(x)
        return x
    return block


# model takes real values vector of size input_dim and via upsampling,
# reshaping, and various convolutional filters generates an image
def Generator(input_shape, init_side=7, init_filters=128,
              num_conv_blocks=4, hidden_dim=1024):
    model_input = Input(shape=input_shape)
    x = model_input

    x = Dense(hidden_dim)(x)
    x = LeakyReLU()(x)

    x = Dense(init_side * init_side * init_filters)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Reshape((init_side, init_side, init_filters))(x)

    for i in range(num_conv_blocks):
        x = upscale(init_filters // (1 if i == 0 else (2 ** i)))(x)

    #x = Dense(hidden_dim)(Flatten()(x))
    #x = Dense(4 * 4 * hidden_dim)(x)
    #x = Reshape((4, 4, hidden_dim))(x)
    #x = upscale(hidden_dim // 2)(x)
    x = Conv2D(3, kernel_size=3, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    return Model(model_input, x)


def Discriminator(input_shape, init_filters=32, num_conv_blocks=3):
    model_input = Input(shape=input_shape)

    x = model_input
    for i in range(num_conv_blocks):
        x = conv(init_filters * (2 ** i))(x)

    features = Flatten()(x)

    output = Dense(1, activation='linear')(features)

    return Model(inputs=[model_input], outputs=output)


def get_gan(models_path: str=None):
    input_shape = (100, )
    img_shape = (56, 56, 3)
    num_conv_blocks = 3
    init_side = 7
    init_filters = 64

    # output check
    side = init_side * (2 ** num_conv_blocks)
    output_shape = (side, side, 3)
    print("Generator out shape = " + str(output_shape))

    optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

    # init GAN components
    generator = Generator(input_shape, init_side=init_side,
                                init_filters=init_filters,
                               num_conv_blocks=num_conv_blocks)
    assert output_shape == generator.output_shape[1:]
    assert img_shape == generator.output_shape[1:]

    discriminator = Discriminator(output_shape)

    # compile discriminator
    d_optimizer = Adam(lr=0.001)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer)

    # build adversarial model
    x = Input(shape=input_shape)

    gan = Model(x, discriminator(generator(x)))
    gan_optimizer = Adam(lr=0.0001)
    gan.compile(loss='binary_crossentropy', optimizer=gan_optimizer)


def get_training_images(images_path, img_size=(64, 64)):
    images = plot_utils.load_data(plot_utils.get_imgs_paths(images_path),
                                  img_size)

    images = images.astype('float32') / 255.

    return images
