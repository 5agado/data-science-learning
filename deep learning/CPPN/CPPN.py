import numpy as np
import logging
import sys
import math
from keras.initializers import RandomNormal
from keras.layers import Activation
from keras.layers import Input, Dense, RepeatVector, Lambda
from keras.layers import add, concatenate
from keras.models import Model

# Based on https://github.com/hardmaru/cppn-tensorflow


class CPPN:
    # TODO try Python 3.7 data class
    def __init__(self, img_width: int, img_height: int, img_depth: int,
                 hidden_dim: int = 32,
                 batch_size: int = 1,
                 nb_hidden_layers: int = 3,
                 latent_dim: int = 32,
                 nb_channels: int = 1,
                 scale_factor: float = 1.0,
                 kernel_init_mean: float = 0.,
                 kernel_init_stddev: float = 1.,
                 inner_activation: str = 'tanh',
                 inner_architecture_key: str = 'base',
                 **kargs):

        # image has 3D coordinates (width, height and depth)
        self.img_width = img_width
        self.img_height = img_height
        self.img_depth = img_depth
        self.nb_points = img_height * img_width * img_depth
        self.img_size = (img_height, img_width, img_depth)
        self.img_shape = (img_height, img_width, img_depth, nb_channels)

        self.hidden_dim = hidden_dim
        self.nb_hidden_layers = nb_hidden_layers
        self.latent_dim = latent_dim
        self.nb_channels = nb_channels
        self.scale_factor = scale_factor

        self.batch_size = batch_size

        self.kernel_init_mean = kernel_init_mean
        # low values generate just gradients
        self.kernel_init_stddev = kernel_init_stddev

        self.inner_activation = inner_activation
        self.inner_architecture_key = inner_architecture_key

        self.model = self._create_model()

    def _get_kernel_initializer(self):
        """
        Provide kernel initializer for model layers.
        This is a critical factor for obtaining interesting results; the initial model will otherwise produce
        boring gradients
        :return:
        """
        mean = self.kernel_init_mean
        stddev = self.kernel_init_stddev

        #initializer = VarianceScaling(scale=stddev)
        initializer = RandomNormal(mean=mean, stddev=stddev)
        return initializer

    def _create_model(self):
        """
        Create CPPN Keras Model
        :return:
        """
        kernel_initializer = self._get_kernel_initializer()

        # Inputs
        x_vec = Input(shape=(self.nb_points, 1), name='in_x')
        y_vec = Input(shape=(self.nb_points, 1), name='in_y')
        z_vec = Input(shape=(self.nb_points, 1), name='in_z')
        r_vec = Input(shape=(self.nb_points, 1), name='in_r')
        e_vec = Input(shape=(self.nb_points, 1), name='in_e')
        latent_vec = Input(shape=(self.latent_dim,), name='in_latent')

        # Repeat z for each point and scale
        latent_unroll = RepeatVector(self.nb_points)(latent_vec)
        # TODO compare scaling of z with scaling data
        latent_scaled = Lambda(lambda x: x * self.scale_factor)(latent_unroll)

        # Input dense
        x = concatenate([
            Dense(units=self.hidden_dim, use_bias=False, kernel_initializer=kernel_initializer)(x_vec),
            Dense(units=self.hidden_dim, use_bias=False, kernel_initializer=kernel_initializer)(y_vec),
            Dense(units=self.hidden_dim, use_bias=False, kernel_initializer=kernel_initializer)(r_vec),
            Dense(units=self.hidden_dim, use_bias=False, kernel_initializer=kernel_initializer)(e_vec),
            Dense(units=self.hidden_dim, kernel_initializer=kernel_initializer,
                  bias_initializer=kernel_initializer,
                  )(latent_scaled)
        ])

        x = Activation(self.inner_activation)(x)

        # Configurable internal architecture
        output = self._get_internal_output('base', x)

        # Build model
        cppn = Model([x_vec, y_vec, z_vec, r_vec, e_vec, latent_vec], output)

        return cppn

    def _get_internal_output(self, architecture_key: str, input):
        """
        List of test architecture to try for the CPPN.
        :param architecture_key:
        :param input: input tensor to feed to the architecture
        :return: output tensor from the chosen architecture
        """
        kernel_initializer = self._get_kernel_initializer()

        x = input

        # N fully connected layers with the given activation function
        if architecture_key == 'base':
            for i in range(self.nb_hidden_layers):
                x = Dense(units=self.hidden_dim, kernel_initializer=kernel_initializer,
                          bias_initializer=kernel_initializer,
                          activation=self.inner_activation,
                          name='hidden_{}'.format(i))(x)

            output = Dense(units=self.nb_channels, kernel_initializer=kernel_initializer,
                           activation='sigmoid', name='output')(x)
        # Nx2 fully connected layers with mix of softplus and given activation function
        elif architecture_key == 'softplus':
            for i in range(self.nb_hidden_layers):
                x = Dense(units=self.hidden_dim, kernel_initializer=kernel_initializer,
                          bias_initializer=kernel_initializer,
                          activation='softplus',
                          name='hidden_{}'.format(i))(x)
                x = Dense(units=self.hidden_dim, kernel_initializer=kernel_initializer,
                          bias_initializer=kernel_initializer,
                          activation=self.inner_activation,
                          name='hidden_{}'.format(i))(x)

            output = Dense(units=self.nb_channels, kernel_initializer=kernel_initializer,
                           activation='sigmoid', name='output')(x)
        # N residual layers
        elif architecture_key == 'residual':
            for i in range(self.nb_hidden_layers):
                inner_x = Dense(units=self.hidden_dim, kernel_initializer=kernel_initializer,
                                bias_initializer=kernel_initializer,
                                activation=self.inner_activation,
                                name='hidden_{}'.format(i))(x)
                x = add([x, inner_x])
            output = Dense(units=self.nb_channels, kernel_initializer=kernel_initializer,
                           activation='sigmoid', name='output')(x)
        else:
            logging.error("No such architecture key: {}. Exiting".format(architecture_key))
            sys.exit()

        return output

    def get_data(self, scale: float = 1., translation: float = 0., rotation: float = 0.,
                 extra_fun=lambda x, y, z: x*y*z):
        """
        Get data to feed to the model, based on the batch_size and img_size defined initially.
        For pixel coordinates, values are scaled down in the range [-1, 1].
        r is the distance from the center for each coordinate
        :param extra_fun: function to compute an extra matrix data from x and y values
        :param scale:
        :param translation:
        :param rotation: rotation factor in degrees
        :return: x, y, r and the extra vector
        """
        # get pixels coordinates in the range [-1, 1]
        # this would be equivalent to explicitly operating min-max normalization
        x_range = scale * np.linspace(-1., 1., self.img_width) + translation
        y_range = scale * np.linspace(-1., 1., self.img_height) + translation
        z_range = scale * np.linspace(-1., 1., self.img_depth) + translation

        # repeat each range along the opposite axis
        x_mat, y_mat, z_mat = np.meshgrid(x_range, y_range, z_range)

        # TODO do 3d rotation
        # if give, rotate coords via rotation matrix
        if rotation != 0:
            theta = math.radians(rotation)
            rot_matrix = np.array([[math.cos(theta), -math.sin(theta)],
                                   [math.sin(theta), math.cos(theta)]])
            new_mat = np.transpose([x_mat, y_mat]) @ rot_matrix
            x_mat = new_mat.transpose()[0]
            y_mat = new_mat.transpose()[1]

        # compute radius matrix
        r_mat = np.sqrt(x_mat * x_mat + y_mat * y_mat + z_mat * z_mat)

        # computer additional arbitrary matrix based on given key
        extra_mat = extra_fun(x_mat, y_mat, z_mat)

        # flatten matrices and reshape based on batch size
        x = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, self.nb_points, 1)
        y = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, self.nb_points, 1)
        z = np.tile(z_mat.flatten(), self.batch_size).reshape(self.batch_size, self.nb_points, 1)
        r = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, self.nb_points, 1)
        extra = np.tile(extra_mat.flatten(), self.batch_size).reshape(self.batch_size, self.nb_points, 1)

        return x, y, z, r, extra

    def get_latent(self):
        """
        Sample batch_size random vectors from uniform distribution
        :return:
        """
        latent_vec = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.latent_dim)).astype(np.float32)
        return latent_vec

    def generate_imgs(self, x, y, z, r, e, latent):
        """
        Generate batch_size images feeding the given data to the model.
        :param x:
        :param y:
        :param z:
        :param r:
        :param e:
        :param latent:
        :return: batch_size images already reshaped to be displayed or saved
        """
        prediction = self.model.predict([x, y, z, r, e, latent])
        if self.nb_channels == 1:
            return prediction.reshape([self.batch_size, *self.img_size])
        else:
            return prediction.reshape([self.batch_size, *self.img_shape])
