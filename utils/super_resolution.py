import math
from ast import literal_eval

from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

from utils import image_processing


def conv(filters, kernel_size=5, strides=2, leaky_relu=False):
    def block(x):
        x = Conv2D(filters, kernel_size=kernel_size, strides=strides,
                   padding='same')(x)
        x = BatchNormalization()(x)
        if leaky_relu:
            x = LeakyReLU(0.2)(x)
        else:
            x = Activation("relu")(x)
        return x
    return block


def res_block(filters, kernel_size=3):
    def block(input_tensor):
        x = conv(filters, kernel_size=kernel_size, strides=1, leaky_relu=True)(input_tensor)
        x = conv(filters, kernel_size=kernel_size, strides=1, leaky_relu=True)(input_tensor)
        x = Add()([x, input_tensor])
        x = LeakyReLU(alpha=0.2)(x)
        return x
    return block


def upscale(filters, kernel_size=3):
    def block(x):
        x = UpSampling2D()(x)
        x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        return x
    return block


def get_upsampling_model(input_shape, upscale_factor=2):
    # get number of upscale blocks needed based on given factor
    # (2^upscale_blocks = upscale_factor)
    upscale_blocks = int(math.log(upscale_factor, 2))

    model_input = Input(shape=input_shape)
    x = model_input

    res_input_0 = conv(64, 3, 1)(x)

    x = res_block(64)(res_input_0)

    nb_residual = 5
    for i in range(nb_residual):
        x = res_block(64)(x)

    x = Add()([x, res_input_0])

    # if more should get input for Nth conv block and add for Nth res block
    for i in range(upscale_blocks):
        x = upscale(64, 3)(x)
        # first res_input is just model_input
        # x = Add()([x, res_inputs[i+2]])

    outp = Conv2D(3, 3, activation='linear', padding='same')(x)

    return model_input, outp


def get_SRResNet(cfg):
    models_path = cfg.get('models_path', None)
    LR_IMG_SHAPE = literal_eval(cfg.get('LR_IMG_SHAPE'))

    upsampled_inp, upsampled_output = get_upsampling_model(LR_IMG_SHAPE, 2)

    sr_model = Model(upsampled_inp, upsampled_output)
    #adam = Adam(lr=1e-3)
    #sr_model.compile(optimizer=adam, loss=loss_wrapper(vgg_hr, vgg_upsamples), metrics=[PSNRLoss])

    if models_path:
        print("Loading Super Resolution Models...")
        sr_model.load_weights(models_path + '/sr_model.h5')
        print("Super Resolution Models Loaded")

    return sr_model


################
# Training Utils
################
def train_generator(imgs_filepaths, img_size=None, batch_size=64, apply_fun=None):
    while True:
        imgs_paths_sample = np.random.choice(imgs_filepaths, batch_size)
        original_data = image_processing.load_data(imgs_paths_sample, img_size)
        original_data = original_data / 255.0
        if apply_fun:
            data = apply_fun(original_data)
        else:
            data = original_data
        yield data, original_data