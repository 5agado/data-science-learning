import os
import math
from tqdm import tqdm
import numpy as np
from typing import List
from tensorflow.python.keras.layers import Add, Layer, Conv2D, Conv2DTranspose, InputLayer
from tensorflow.python.keras.layers import Dense, Flatten, Input, Reshape
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.layers import LeakyReLU, BatchNormalization, Lambda, Activation
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import VarianceScaling
from tensorflow.python.keras.constraints import max_norm
import tensorflow as tf

# Based on
# * https://github.com/NVlabs/stylegan
# * https://github.com/ialhashim/StyleGAN-Tensorflow2/blob/master/stylegan.py


class StyleGan:
    def __init__(self, config):
        self.config = config
        self.config['model']['generator']['z_size'] = self.config['data']['z_size']
        self.config['model']['discriminator']['z_size'] = self.config['data']['z_size']

        self._build_model()

    def _build_model(self):
        self.generator = StyleGAN_G(self.config['model']['generator'])

        self.discriminator = StyleGAN_D(self.config['model']['discriminator'])

    @staticmethod
    # maps noise/latent vector to style vector
    def _build_mapping_network(config):
        latent_vector = Input(config['z_size'], name='mapping/latent_vector')

        # normalize latent vector
        x = PixelNorm(name='mapping/pixel_norm')(latent_vector)

        # Mapping layers.
        for layer_idx in range(config['nb_mapping_layers']):
            name = f'mapping/dense_{layer_idx}'

            x = DenseLayer(units=config['dlatent_size'] if layer_idx == config['nb_mapping_layers'] - 1 else config['mapping_fmaps'],
                           kernel_initializer=GetWeights(), name=name, lrmul=config['mapping_lrmul'])(x)
            x = LeakyReLU(alpha=0.2)(x)

        # Broadcast.
        x = Broadcast(name='mapping/broadcast')(x)

        # Output.
        x = Identity(name='mapping/dlatents_out')(x)

        # Apply truncation trick.
        model_output = Truncation(name='mapping/truncation')(x)

        model = Model(latent_vector, model_output)

        return model

    @staticmethod
    def _build_synthesis_network(config):
        num_channels = 3
        resolution_log2 = int(np.log2(config['resolution']))
        num_layers = resolution_log2 * 2 - 2
        num_styles = num_layers

        # inputs
        dlatents = Input([num_styles, config['dlatent_size']], name='synthesis/dlatents')

        # Noise inputs.
        noise_inputs = []
        for layer_idx in range(num_layers):
            noise_inputs.append(RandomNoise(name=f'synthesis/noise_{layer_idx}', layer_idx=layer_idx)(dlatents))

        # Things to do at the end of each layer.
        def layer_epilogue(x, layer_idx, name):
            name = 'G_synthesis/{}x{}/{}/'.format(x.shape[2], x.shape[2], name)

            x = ApplyNoise(name=name + 'Noise')([x, noise_inputs[layer_idx]])
            x = ApplyBias(name=name + 'bias')(x)
            x = LeakyReLU(alpha=0.2, name=name + 'LeakyReLU')(x)
            x = InstanceNorm(name=name + 'InstanceNorm')(x)

            style = DenseLayer(units=x.shape[1] * 2, gain=1, name=name + 'StyleMod')(
                StridedSlice(layer_idx, name=name + 'StridedSlice')(dlatents))
            x = StyleModApply(name=name + 'StyleModApply')([x, style])

            return x

        # Building blocks for remaining layers.
        def block(res, x):  # res = 3..resolution_log2
            name, name0, name1 = '%dx%d' % (2 ** res, 2 ** res), 'Conv0_up', 'Conv1'

            # Conv0_up
            upscaled = Upscale2d_conv2d(x, name='G_synthesis/{}/{}'.format(name, name0), filters=nf(res - 1),
                                        kernel_size=3, use_bias=False)
            x = layer_epilogue(Blur(name='G_synthesis/{}/{}/Blur'.format(name, name0))(upscaled), res * 2 - 4, name0)

            # Conv1
            x = layer_epilogue(Conv2d(name='G_synthesis/{}/{}'.format(name, name1), filters=nf(res - 1), kernel_size=3,
                                      use_bias=False)(x), res * 2 - 3, name1)

            return x

        def torgb(res, x):  # res = 2..resolution_log2
            lod = resolution_log2 - res
            return Conv2d(name='G_synthesis/ToRGB_lod%d' % lod, filters=num_channels, kernel_size=1, gain=1,
                          use_bias=True)(x)

        # Early layers.
        x = layer_epilogue(Const(name='G_synthesis/4x4/Const')(dlatents), 0, name='Const')
        x = layer_epilogue(Conv2d(name='G_synthesis/4x4/Conv', filters=nf(1), kernel_size=3, use_bias=False)(x), 1,
                           'Conv')

        # Fixed structure: simple and efficient, but does not support progressive growing.
        for res in range(3, resolution_log2 + 1):
            x = block(res, x)

        x = torgb(resolution_log2, x)

        return Model(inputs=dlatents, outputs=x, name='G_synthesis')

    def train(self, train_ds, validation_ds, nb_epochs: List[int], log_dir, checkpoint_dir, is_tfdataset=False,
               restore_latest_checkpoint=True):
        raise NotImplementedError

    def _train_step(self, gen, dis, gan, dataset, nb_epochs, step,
                    plot_summary_writer, train_summary_writer,
                    fadein=False):
        raise NotImplementedError

    def setup_dataset(self, dataset):
        # prefetch lets the dataset fetch batches in the background while the model is training
        return dataset.shuffle(self.config['data']['buffer_size']) \
                            .batch(self.config['training']['batch_size'], drop_remainder=True) \
                            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    @staticmethod
    def get_optimizer(config):
        return Adam(lr=config['learning_rate'], beta_1=config['beta_1'], beta_2=config['beta_2'],
                    epsilon=config['epsilon'])


class StyleGAN_G(Model):
    def __init__(self, config):
        super(StyleGAN_G, self).__init__()
        self.model_mapping = StyleGan._build_mapping_network(config)
        self.model_synthesis = StyleGan._build_synthesis_network(config)
        print('Model created.')

    def call(self, inputs):
        x = self.model_mapping(inputs)
        x = self.model_synthesis(x)
        return x

    def generate_sample(self, seed=5, is_visualize=False):
        rnd = np.random.RandomState(seed)
        latents = rnd.randn(1, 512)

        y = self.predict(latents)

        images = y.transpose([0, 2, 3, 1])
        images = np.clip((images + 1) * 0.5, 0, 1)

        if is_visualize:
            print(images.shape, np.min(images), np.max(images))

            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 10))
            plt.imshow(images[0])
            plt.show()

        return images


class StyleGAN_D(Model):
    def __init__(self, config):
        super(StyleGAN_D, self).__init__()

        resolution = config['resolution']
        mbstd_group_size = config['mbstd_group_size']
        mbstd_num_features = config['mbstd_num_features']

        resolution_log2 = int(math.log2(resolution))

        model = Sequential(name='Discriminator')
        model.add(InputLayer(input_shape=[3, resolution, resolution]))

        def fromrgb(res):
            name = 'FromRGB_lod%d' % (resolution_log2 - res)
            model.add( Conv2d(filters=nf(res-1), kernel_size=1, name=name) )
            model.add( LeakyReLU(alpha=0.2, name=name+'/LeakyReLU') )

        def block(res):
            name = '%dx%d' % (2**res, 2**res)
            if res >= 3: # 8x8 and up
                model.add( Conv2d(filters=nf(res-1), kernel_size=3, name=name+'/Conv0') )
                model.add( LeakyReLU(alpha=0.2, name=name+'/Conv0/LeakyReLU') )

                model.add( Blur(name=name+'/Blur') )
                Conv2d_downscale2d(model=model, filters=nf(res-2), kernel_size=3, name=name+'/Conv1_down')
                model.add( LeakyReLU(alpha=0.2, name=name+'/Conv1_down/LeakyReLU') )

            else: # 4x4
                if mbstd_group_size > 1:
                    model.add( Lambda(lambda x: minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features), name=name+'/MinibatchStddev') )

                model.add( Conv2d(filters=nf(res-1), kernel_size=3, name=name+'/Conv') )
                model.add( LeakyReLU(alpha=0.2, name=name+'/Conv/LeakyReLU') )

                model.add( Flatten() )
                model.add( DenseLayer(units=nf(res-2), kernel_initializer=GetWeights(), name=name+'/Dense0') )
                model.add( LeakyReLU(alpha=0.2, name=name+'/Dense0/LeakyReLU') )

                model.add( DenseLayer(units=1, kernel_initializer=GetWeights(1), gain=1, name=name+'/Dense1') )

        # Blocks
        fromrgb(resolution_log2)
        for res in range(resolution_log2, 2, -1): block(res)
        block(2)

        self.model = model

    def call(self, inputs):
        return self.model(inputs)


def nf(stage, fmap_base=8192, fmap_decay=1.0, fmap_max=512):
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)


def GetWeights(gain=math.sqrt(2)):
    return VarianceScaling(gain)


def runtime_coef(kernel_size, gain, fmaps_in, fmaps_out, lrmul=1.0):
    # Equalized learning rate and custom learning rate multiplier.
    shape = [kernel_size[0], kernel_size[1], fmaps_in, fmaps_out]
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init
    init_std = 1.0 / lrmul
    return he_std * lrmul

# TODO validate how different is this dense layer compared to base one
class DenseLayer(Dense):
    def __init__(self, units, name, kernel_initializer=GetWeights(), gain=math.sqrt(2), lrmul=1.0):
        super(DenseLayer, self).__init__(units=units, kernel_initializer=kernel_initializer, name=name)
        self.gain = gain
        self.lrmul = lrmul

    def call(self, inputs):
        x, b, w = inputs, self.bias * self.lrmul, self.kernel * runtime_coef([1, 1], self.gain, inputs.shape[1],
                                                                             self.units, lrmul=self.lrmul)

        # Input x kernel
        if len(x.shape) > 2: x = tf.reshape(x, [-1, np.prod([d for d in x.shape[1:]])])
        x = tf.matmul(x, w)

        # Bias
        if len(x.shape) == 2:
            return x + b

        return x + tf.reshape(b, [1, -1, 1, 1])


class Conv2d(Conv2D):
    def __init__(self, filters, kernel_size, name, gain=math.sqrt(2), lrmul=1.0, kernel_modifier=None, strides=1,
                 use_bias=True):
        super(Conv2d, self).__init__(filters=filters, kernel_size=kernel_size, kernel_initializer=GetWeights(gain),
                                     use_bias=use_bias, padding='same', data_format='channels_first', name=name,
                                     strides=strides)
        self.gain = gain
        self.lrmul = lrmul
        self.kernel_modifier = kernel_modifier

    # Perform convolution with modified kernel then add bias
    def call(self, inputs):
        if self.kernel_modifier is None:
            w = self.kernel
        else:
            w = self.kernel_modifier(self.kernel)

        outputs = self._convolution_op(inputs, w * runtime_coef(self.kernel_size, self.gain, inputs.shape[1],
                                                                self.filters))

        if self.use_bias:
            b = self.bias * self.lrmul
            if self.data_format == 'channels_first':
                outputs = tf.nn.bias_add(outputs, b, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(outputs, b, data_format='NHWC')

        return outputs


class Const(Layer):
    def __init__(self, name):
        super(Const, self).__init__(name=name)

    def build(self, input_shape):
        self.const = self.add_variable('const', shape=[1, 512, 4, 4])

    def call(self, inputs):
        return tf.tile(self.const, [tf.shape(inputs)[0], 1, 1, 1])


class RandomNoise(Layer):
    def __init__(self, name, layer_idx):
        super(RandomNoise, self).__init__(name=name)

        res = layer_idx // 2 + 2
        self.layer_idx = layer_idx
        self.noise_shape = [1, 1, 2 ** res, 2 ** res]

    def build(self, input_shape):
        self.noise = self.add_variable('noise', shape=self.noise_shape, initializer=tf.initializers.zeros(),
                                       trainable=False)

    def call(self, inputs):
        return self.noise


class ApplyNoise(Layer):
    def __init__(self, name):
        super(ApplyNoise, self).__init__(name=name)

    def build(self, input_shape):
        input_shape = input_shape[0]
        self.weight = self.add_variable('weight', shape=[input_shape[1]], initializer=tf.initializers.zeros())

    def call(self, inputs):
        # noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        x, noise = inputs

        return x + noise * tf.reshape(self.weight, [1, -1, 1, 1])


class ApplyBias(Layer):
    def __init__(self, name, lrmul=1.0):
        super(ApplyBias, self).__init__(name=name)
        self.lrmul = lrmul

    def build(self, input_shape):
        self.bias = self.add_variable('bias', shape=[input_shape[1]])

    def call(self, x):
        b = self.bias * self.lrmul
        if len(x.shape) == 2: return x + b
        return x + tf.reshape(b, [1, -1, 1, 1])


class StridedSlice(Layer):
    def __init__(self, layer_idx, name):
        super(StridedSlice, self).__init__(name=name)
        self.layer_idx = layer_idx

    def call(self, inputs):
        return inputs[:, self.layer_idx]


class StyleModApply(Layer):
    def __init__(self, name):
        super(StyleModApply, self).__init__(name=name)

    def call(self, inputs):
        x, style = inputs

        style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
        return x * (style[:, 0] + 1) + style[:, 1]


def _blur2d(x, f=[1, 2, 1], normalize=True, flip=False, stride=1):
    assert x.shape.ndims == 4 and all(dim is not None for dim in x.shape[1:])
    assert isinstance(stride, int) and stride >= 1

    # Finalize filter kernel.
    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[:, :, np.newaxis, np.newaxis]
    f = np.tile(f, [1, 1, int(x.shape[1]), 1])

    # No-op => early exit.
    if f.shape == (1, 1) and f[0, 0] == 1:
        return x

    # Convolve using depthwise_conv2d.
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d() doesn't support fp16
    f = tf.constant(f, dtype=x.dtype, name='filter')
    # TODO changed NCHW to NHWC and [1, 1, 2, 2]
    strides = [1, stride, stride, 1]
    x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding='SAME', data_format='NHWC')
    x = tf.cast(x, orig_dtype)
    return x


def Blur(name, blur_filter=[1, 2, 1]):
    def blur2d(x, f=[1, 2, 1], normalize=True):
        return _blur2d(x, f, normalize)

    return Lambda(lambda x: blur2d(x, blur_filter), name=name)


def _downscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 2x2, float32 => downscale using _blur2d().
    if factor == 2 and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return _blur2d(x, f=f, normalize=False, stride=factor)

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Large factor => downscale using tf.nn.avg_pool().
    # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
    # TODO changed NCHW to NHWC and [1, 1, 2, 2]
    ksize = [1, factor, factor, 1]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NHWC')


def _upscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Upscale using tf.tile().
    s = x.shape
    x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = tf.tile(x, [1, 1, 1, factor, 1, factor])
    x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x


def Downscaled2d(name, factor=2, gain=1):
    return Lambda(lambda x: _downscale2d(x, factor, gain), name=name + '/Downscaled2d')


def Upscaled2d(name, factor=2, gain=1):
    return Lambda(lambda x: _upscale2d(x, factor, gain), name=name + '/Upscaled2d')


def Conv2d_downscale2d(model, filters, kernel_size, name, gain=math.sqrt(2), fused_scale='auto'):
    if fused_scale == 'auto':
        x = model.layers[-1].output
        fused_scale = min(x.shape[2:]) >= 128

    if not fused_scale:
        # Not fused => call the individual ops directly.
        model.add(Conv2d(filters, kernel_size, name, gain))
        model.add(Downscaled2d(name))
    else:
        # Fused => perform both ops simultaneously using tf.nn.conv2d().
        def fused_op(w):
            w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
            w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
            return w

        model.add(Conv2d(filters, kernel_size, name, gain, kernel_modifier=fused_op, strides=2))


def Upscale2d_conv2d(x, filters, kernel_size, name, use_bias, gain=math.sqrt(2), fused_scale='auto'):
    if fused_scale == 'auto':
        fused_scale = min(x.shape[2:]) * 2 >= 128

    if not fused_scale:
        x = Upscaled2d(name)(x)
        x = Conv2d(filters, kernel_size, name=name, gain=gain, use_bias=use_bias)(x)
        return x

    #return Conv2DTranspose(filters, kernel_size, strides=2)(x)
    return Conv2d_transpose(filters, kernel_size, name, gain, strides=2)(x)


class Conv2d_transpose(Conv2DTranspose):
    def __init__(self, filters, kernel_size, name, gain=math.sqrt(2), lrmul=1.0, kernel_modifier=None, strides=2,
                 use_bias=False):
        super(Conv2d_transpose, self).__init__(filters=filters, kernel_size=kernel_size,
                                               kernel_initializer=GetWeights(gain),
                                               use_bias=use_bias, padding='same', data_format='channels_first',
                                               name=name, strides=strides)
        self.gain = gain
        self.lrmul = lrmul
        self.kernel_modifier = kernel_modifier

    def build(self, input_shape):
        shape = [self.kernel_size[0], self.kernel_size[1], input_shape[1], self.filters]
        self.kernel = self.add_variable('weight', shape=shape, initializer=tf.initializers.zeros())

    def call(self, inputs):
        # Fused => perform both ops simultaneously using tf.nn.conv2d_transpose().
        def fused_op(w):
            w = tf.transpose(w, [0, 1, 3, 2])  # [kernel, kernel, fmaps_out, fmaps_in]
            w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
            w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
            return w

        x, w = inputs, fused_op(
            self.kernel * runtime_coef(self.kernel_size, self.gain, inputs.shape[1], self.filters,
                                       lrmul=self.lrmul))

        os = [tf.shape(inputs)[0], self.filters, inputs.shape[2] * 2, inputs.shape[3] * 2]

        # TODO changed NCHW to NHWC and [1, 1, 2, 2]
        outputs = tf.nn.conv2d_transpose(x, w, os, strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC')

        return outputs


def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    group_size = tf.minimum(group_size,
                            tf.shape(x)[0])  # Minibatch must be divisible by (or smaller than) group_size.
    s = x.shape  # [NCHW]  Input shape.
    y = tf.reshape(x, [group_size, -1, num_new_features, s[1] // num_new_features, s[2], s[
        3]])  # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
    y = tf.cast(y, tf.float32)  # [GMncHW] Cast to FP32.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)  # [GMncHW] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)  # [MncHW]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)  # [MncHW]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)  # [Mn111]  Take average over fmaps and pixels.
    y = tf.reduce_mean(y, axis=[2])  # [Mn11] Split channels into c channel groups
    y = tf.cast(y, x.dtype)  # [Mn11]  Cast back to original data type.
    y = tf.tile(y, [group_size, 1, s[2], s[3]])  # [NnHW]  Replicate over group and pixels.
    return tf.concat([x, y], axis=1)  # [NCHW]  Append as new fmap.


class PixelNorm(Layer):
    def __init__(self, name):
        """
        Provide activations statistical summary
        """
        super(PixelNorm, self).__init__(name=name)

    def call(self, inputs, **kwargs):
        # normalize values by the sqrt of the mean squared (L2 norm)
        return inputs / K.sqrt(K.mean(K.square(inputs), axis=1, keepdims=True) + 1.0e-8) # ??reduce_mean VS mean


class InstanceNorm(Layer):
    def __init__(self, name):
        super(InstanceNorm, self).__init__(name=name)

    def call(self, inputs, **kwargs):
        orig_dtype = inputs.dtype
        x = K.cast(inputs, tf.float32)
        x -= K.mean(x, axis=[2, 3], keepdims=True)
        x /= K.sqrt(K.mean(K.square(x), axis=[2, 3], keepdims=True) + 1e-8)
        x = K.cast(x, orig_dtype)
        return x


def Identity(name):
    return Lambda(lambda x: x, name=name)


def Broadcast(name, dlatent_broadcast=18):
    def broadcast(x):
        return K.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])

    return Lambda(lambda x: broadcast(x), name=name)


class Truncation(Layer):
    def __init__(self, name, num_layers=18, truncation_psi=0.7, truncation_cutoff=8):
        super(Truncation, self).__init__(name=name)
        self.num_layers = num_layers
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

    def build(self, input_shape):
        self.dlatent_avg = self.add_variable('dlatent_avg', shape=[int(input_shape[-1])])

    def call(self, inputs, **kwargs):
        layer_idx = np.arange(self.num_layers)[np.newaxis, :, np.newaxis]
        ones = np.ones(layer_idx.shape, dtype=np.float32)
        coefs = tf.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)

        def lerp(a, b, t): return a + (b - a) * t

        return lerp(self.dlatent_avg, inputs, coefs)
