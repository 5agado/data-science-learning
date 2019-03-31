from keras.models import *
from keras.layers import *
from keras.layers.core import Activation, Dense
from keras import backend as K
from keras import objectives


# standard convolution block used in the encoder
def encoder_conv_block(filters, block_input, kernel_size=(3, 3), strides=(1, 1)):
    block = Convolution2D(filters, kernel_size, strides=strides, padding='same')(block_input)
    block = LeakyReLU()(block)
    return block


# takes an image and generates two vectors: means and standards deviations
def encoder_model(input_shape, latent_dim, init_filters=64, num_conv_blocks=2):
    input_image = Input(shape=input_shape)

    x = input_image
    for i in range(num_conv_blocks):
        x = encoder_conv_block(init_filters * (2 ** i), block_input=x)

    features = Flatten()(x)

    # gaussian parameters
    mean_vector = Dense(latent_dim, activation='linear')(features)
    std_vector = Dense(latent_dim, activation='linear')(features)

    return Model(inputs=[input_image], outputs=[mean_vector, std_vector])


# utility for the standard deconvolution block used in the decoder
def decoder_deconv_block(filters, block_input, kernel_size=(3, 3), strides=(1, 1)):
    block = UpSampling2D()(block_input)
    block = Convolution2D(filters, (3, 3), strides=strides, padding='same')(block)
    block = LeakyReLU()(block)
    block = BatchNormalization()(block)
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


# takes as input the latent vector z
def decoder_model(latent_dim, img_shape, init_filters=128, num_deconv_blocks=2):
    latent_vector = Input([latent_dim])
    init_shape = tuple([get_initial_size(d, num_deconv_blocks)
                        for d in img_shape[:-1]] + [init_filters])

    # CNN part
    x = Dense(1024)(latent_vector)
    x = LeakyReLU()(x)

    x = Dense(np.prod(init_shape))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Reshape(init_shape)(x)

    for i in range(num_deconv_blocks):
        x = decoder_deconv_block(init_filters // (1 if i == 0 else (2 ** i)), block_input=x)

    x = Convolution2D(img_shape[-1], (2, 2), padding='same', activation='sigmoid')(x)

    return Model(inputs=latent_vector, outputs=x)


def get_vae_model(img_shape, latent_dim, init_filters=128, batch_size=32):
    # init model components
    encoder = encoder_model(input_shape=img_shape, latent_dim=latent_dim, init_filters=init_filters)
    decoder = decoder_model(latent_dim=latent_dim, img_shape=img_shape, init_filters=init_filters)

    # Build model
    input_img = Input(shape=(img_shape))
    mean_vector, std_vector = encoder(inputs=input_img)

    latent_vector = Lambda(lambda x: sampling(x[0], x[1], batch_size,
                                              latent_dim))([mean_vector, std_vector])

    output_img = decoder(latent_vector)

    # Compile model

    # for the loss consider the sum of the generative loss
    # and the latent loss (KL divergence)
    def vae_loss(real_image, generated_image):
        gen_loss = K.mean(objectives.mean_squared_error(real_image, generated_image))
        kl_loss = - 0.5 * K.mean(1 + std_vector - K.square(mean_vector) - K.exp(std_vector), axis=-1)
        # kl_loss = 0.5 * K.mean(K.square(std_vector) + K.square(mean_vector) - K.log(K.square(std_vector)) -1, axis=-1)
        return gen_loss + kl_loss

    vae = Model(inputs=input_img, outputs=output_img)

    # TODO
    # Having/tracking separate loss functions which are then combined by Keras by given weights
    # in order for this to work you also need multiple outputs, one for each loss
    # gen_loss = Lambda(lambda real_image, generated_image : K.mean(objectives.mean_squared_error(real_image, generated_image)))
    # kl_loss = Lambda(lambda _1, _2 : - 0.5 * K.mean(1 + std_vector - K.square(mean_vector) - K.exp(std_vector), axis=-1))
    # vae.compile(loss=[gen_loss, kl_loss], loss_weights=[1., 1.], optimizer="adam")

    vae.compile(loss=vae_loss, optimizer="adam")

    return vae


# sample latent vector using learned distribution parameters
def sampling(z_mean, z_log_sigma, batch_size, latent_dim):
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=1.)
    # ?? use K.exp(z_log_sigma). Isn't our param is already z_sigma??
    return z_mean + z_log_sigma * epsilon