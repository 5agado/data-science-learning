import os
from tqdm import tqdm
import numpy as np
from typing import List
from tensorflow.python.keras.layers import Add, Layer, Conv2D, AveragePooling2D, UpSampling2D, merge
from tensorflow.python.keras.layers import Dense, Flatten, Input, Reshape
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.layers import LeakyReLU, BatchNormalization, Lambda, Activation
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.constraints import max_norm
import tensorflow as tf

# Adapted from https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-keras-for-synthesizing-faces/


class ProGan:
    def __init__(self, config):
        self.config = config
        self.config['model']['generator']['z_size'] = self.config['data']['z_size']
        self.config['model']['generator']['num_blocks'] = self.config['data']['num_blocks']
        self.config['model']['discriminator']['num_blocks'] = self.config['data']['num_blocks']

        self._build_model()

    def _build_model(self):
        # Generator
        self.generators = ProGan._build_generator((4, 4, 3), self.config['model']['generator'])

        # Discriminator
        self.discriminators = ProGan._build_discriminator(self.config['model']['discriminator'])

        # define composite models
        self.gan_models = self._define_composite(self.generators, self.discriminators)

    def _define_composite(self, generators, discriminators):
        model_list = []
        for i in range(len(discriminators)):
            g_models, d_models = generators[i], discriminators[i]
            # straight-through model
            d_models[0].trainable = False
            model1 = Sequential()
            model1.add(g_models[0])
            model1.add(d_models[0])
            model1.compile(loss=wasserstein_loss, optimizer=ProGan.get_optimizer(self.config['model']['generator']))
            # fade-in model
            d_models[1].trainable = False
            model2 = Sequential()
            model2.add(g_models[1])
            model2.add(d_models[1])
            model2.compile(loss=wasserstein_loss, optimizer=ProGan.get_optimizer(self.config['model']['generator']))
            # store
            model_list.append([model1, model2])
        return model_list

    @staticmethod
    # takes an image and generates two vectors: means and standards deviations
    def _build_generator(init_shape, config):
        latent_vector = Input(config['z_size'], name='generator_input')

        x = Dense(np.prod(init_shape))(latent_vector)
        x = Reshape(init_shape)(x)

        # weights init
        w_init = RandomNormal(stddev=0.02)
        w_const = max_norm(1.0)

        # conv layers
        for i, strides in enumerate([4, 3]):
            x = Conv2D(config['filters'], strides, padding='same',
                       kernel_initializer=w_init, kernel_constraint=w_const)(x)
            x = PixelNormalization()(x)
            x = LeakyReLU()(x)

        # conv 1x1
        out_image = Conv2D(config['n_channels'], 1, padding='same')(x)

        model = Model(latent_vector, out_image)

        # store model
        model_list = [[model, model]]
        # create submodels
        for i in range(1, config['num_blocks']):
            # get prior model without the fade-on
            old_model = model_list[i - 1][0]
            # create new model for next resolution
            models = ProGan._add_generator_block(old_model, config)
            # store model
            model_list.append(models)
        return model_list

    @staticmethod
    def _add_generator_block(old_model, config):
        # get the end of the last block
        block_end = old_model.layers[-2].output

        # weights init
        w_init = RandomNormal(stddev=0.02)
        w_const = max_norm(1.0)

        # upsample, and define new block
        upsampling = UpSampling2D()(block_end)

        # conv layers
        x = upsampling
        for i, strides in enumerate([3, 3]):
            # TODO should half filters every new block (and start with more than 128)
            x = Conv2D(config['filters'], strides, padding='same',
                       kernel_initializer=w_init, kernel_constraint=w_const)(x)
            x = PixelNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)

        # add new output layer
        out_image = Conv2D(config['n_channels'], 1, padding='same', activation='tanh')(x)
        # define upsampled straight-through     model
        model1 = Model(old_model.input, out_image)

        # get the output layer from old model
        out_old = old_model.layers[-1]
        # connect the upsampling to the old output layer
        out_image2 = out_old(upsampling)
        # define new output image as the weighted sum of the old and new models
        merged = WeightedSum()([out_image2, out_image])
        # define model
        model2 = Model(old_model.input, merged)

        return [model1, model2]

    @staticmethod
    def _build_discriminator(config):
        model_input = Input(shape=tuple(config['input_shape']), name="base_dis_input")

        # weights init
        w_init = RandomNormal(stddev=0.02)
        w_const = max_norm(1.0)

        # conv layers
        x = model_input
        for i, strides in enumerate([1, 3, 4]):
            # ?? why minibatch layer only after 1x1 conv
            if i == 1:
                x = MinibatchStdev()(x)
            x = Conv2D(config['filters'], strides, padding='same',
                       kernel_initializer=w_init, kernel_constraint=w_const)(x)
            x = LeakyReLU()(x)

        # dense output
        features = Flatten()(x)
        model_output = Dense(1)(features)

        # compile model
        model = Model(model_input, model_output)
        model.compile(loss=wasserstein_loss, optimizer=ProGan.get_optimizer(config))

        # store model
        model_list = [[model, model]]
        # create submodels
        for i in range(1, config['num_blocks']):
            # get prior model without the fade-on
            old_model = model_list[i - 1][0]
            # create new model for next resolution
            models = ProGan._add_discriminator_block(old_model, config)
            # store model
            model_list.append(models)
        return model_list

    @staticmethod
    def _add_discriminator_block(old_model, config):
        # new shape is double the size of previous one
        old_input_shape = list(old_model.input.shape)
        new_input_shape = (old_input_shape[-2]*2, old_input_shape[-2]*2, old_input_shape[-1])
        model_input = Input(shape=new_input_shape, name="doubled_dis_input")

        # weights init
        w_init = RandomNormal(stddev=0.02)
        w_const = max_norm(1.0)

        # conv layers
        x = model_input
        for strides in [1, 3, 3]:
            x = Conv2D(config['filters'], strides, padding='same',
                       kernel_initializer=w_init, kernel_constraint=w_const)(x)
            x = LeakyReLU()(x)

        x = AveragePooling2D()(x)

        new_block = x
        # skip the input, 1x1 and activation for the old model
        for i in range(config['num_input_layers'], len(old_model.layers)):
            x = old_model.layers[i](x)

        # define straight-through model
        model1 = Model(model_input, x)

        # compile model
        model1.compile(loss=wasserstein_loss, optimizer=ProGan.get_optimizer(config))

        # downsample the new larger image
        downsample = AveragePooling2D()(model_input)

        # connect old input processing to downsampled new input
        old_block = old_model.layers[1](downsample)
        old_block = old_model.layers[2](old_block)

        # fade in output of old model input layer with new input
        x = WeightedSum()([old_block, new_block])
        # skip the input, 1x1 and activation for the old model
        for i in range(config['num_input_layers'], len(old_model.layers)):
            x = old_model.layers[i](x)

        # define fade-in model
        model2 = Model(model_input, x)

        # compile model
        model2.compile(loss=wasserstein_loss, optimizer=ProGan.get_optimizer(config))

        return [model1, model2]

    def train(self, train_ds, validation_ds, nb_epochs: List[int], log_dir, checkpoint_dir, is_tfdataset=False,
               restore_latest_checkpoint=True):
        plot_summary_writer = tf.summary.create_file_writer(str(log_dir / 'plot'))
        train_summary_writer = tf.summary.create_file_writer(str(log_dir / 'train'))

        # checkpoints
        if checkpoint_dir:
            raise NotImplementedError

        # baseline model
        gen_normal = self.generators[0][0]
        dis_normal = self.discriminators[0][0]
        gan_normal = self.gan_models[0][0]

        gen_shape = gen_normal.output_shape[1:-1]
        scaled_ds = self.setup_dataset(train_ds.map(lambda img: tf.image.resize(img, gen_shape)))
        self._train_step(gen_normal, dis_normal, gan_normal, scaled_ds, nb_epochs[0], 0,
                         plot_summary_writer, train_summary_writer)

        # train loop
        for i in range(1, len(self.generators)):
            # retrieve models for this level of growth
            [g_normal, g_fadein] = self.generators[i]
            [d_normal, d_fadein] = self.discriminators[i]
            [gan_normal, gan_fadein] = self.gan_models[i]

            # scale dataset to appropriate size
            gen_shape = g_normal.output_shape[1:-1]
            scaled_ds = self.setup_dataset(train_ds.map(lambda img: tf.image.resize(img, gen_shape, 'nearest')))

            # train fade-in models for next level of growth
            self._train_step(g_fadein, d_fadein, gan_fadein, scaled_ds, nb_epochs[i], i,
                             plot_summary_writer, train_summary_writer, True)

            # train normal or straight-through models
            self._train_step(g_normal, d_normal, gan_normal, scaled_ds, nb_epochs[i], i,
                             plot_summary_writer, train_summary_writer, False)

            # checkpoint
            if checkpoint_dir:
                raise NotImplementedError

    def _train_step(self, gen, dis, gan, dataset, nb_epochs, step,
                    plot_summary_writer, train_summary_writer,
                    fadein=False):
        batch_size = self.config['training']['batch_size']
        z_dim = self.config['data']['z_size']

        plot_noise = tf.random.normal([batch_size, z_dim])

        for epoch in tqdm(range(nb_epochs)):
            if fadein:
                update_fadein([gen, dis, gan], epoch, nb_epochs)
            gen_losses = []
            disc_losses = []
            for ds_batch in dataset:
                # train discriminator
                generated_images = gen.predict(tf.random.normal([batch_size, z_dim]))

                d_loss1 = dis.train_on_batch(ds_batch, np.ones((batch_size, 1)))
                d_loss2 = dis.train_on_batch(generated_images, -np.ones((batch_size, 1)))

                # weight clipping to enforce Lipschitz constraint
                clip_threshold = self.config['model']['discriminator']['clip_threshold']
                for l in dis.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -clip_threshold, clip_threshold) for w in weights]
                    l.set_weights(weights)

                # train generator via the discriminator's error
                # ??should GAN train on double batch size given that discriminator
                # trains on batch-size real + batch-size fake
                g_loss = gan.train_on_batch(tf.random.normal([batch_size, z_dim]), np.ones((batch_size, 1)))

                # aggregate losses
                last_batch = ds_batch
                gen_losses.append(g_loss)
                disc_losses.append(np.mean([d_loss1, d_loss2]))

            # Loss summary
            avg_gen_loss = np.mean(gen_losses)
            avg_disc_loss = np.mean(disc_losses)
            with train_summary_writer.as_default():
                tf.summary.scalar("Average Gen Loss {} - {}".format(step, fadein), avg_gen_loss, step=epoch)
                tf.summary.scalar("Average Disc Loss {} - {}".format(step, fadein), avg_disc_loss, step=epoch)

            # Plot data
            with plot_summary_writer.as_default():
                # Plot sample data
                predictions = gen.predict(plot_noise)
                tf.summary.image("Sample Generated {} - {}".format(step, fadein), predictions, step=epoch)
                tf.summary.image("Sample Input {} - {}".format(step, fadein),
                                 [last_batch[np.random.randint(len(last_batch))]], step=epoch)

    def setup_dataset(self, dataset):
        # prefetch lets the dataset fetch batches in the background while the model is training
        return dataset.shuffle(self.config['data']['buffer_size']) \
                            .batch(self.config['training']['batch_size'], drop_remainder=True) \
                            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    @staticmethod
    def get_optimizer(config):
        return Adam(lr=config['learning_rate'], beta_1=config['beta_1'], beta_2=config['beta_2'],
                    epsilon=config['epsilon'])


# update the alpha value on each instance of WeightedSum
def update_fadein(models, step, n_steps):
    # calculate current alpha (linear from 0 to 1)
    alpha = step / float(n_steps - 1)
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                K.set_value(layer.alpha, alpha)


# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, interpolated_samples):
    gradients = K.gradients(y_pred, interpolated_samples)[0]

    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients), axis=[range(1, len(gradients.shape))]))
    gradient_penalty = K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


class WeightedSum(Add):
    def __init__(self, alpha=0.0, **kwargs):
        """
        Used during the growth phase to transition from one image size to the next.
        Alpha is linearly scaled from 0. (full weight to old layer) to 1. (full weight to new layer)
        :param alpha:
        :param kwargs:
        """
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = K.variable(alpha, name='ws_alpha')

    def _merge_function(self, inputs):
        # support only sum of two inputs
        assert (len(inputs) == 2)
        # ((1-a)* input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


class MinibatchStdev(Layer):
    def __init__(self, **kwargs):
        """
        Provide activations statistical summary
        :param kwargs:
        """
        super(MinibatchStdev, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # calculate the mean value for each pixel across channels
        mean = K.mean(inputs, axis=0, keepdims=True)
        # calculate the squared differences between pixel values and mean
        squ_diffs = K.square(inputs - mean)
        # calculate the average of the squared differences (variance)
        mean_sq_diff = K.mean(squ_diffs, axis=0, keepdims=True)
        # add a small value to avoid a blow-up when we calculate stdev
        mean_sq_diff += 1e-8
        # square root of the variance (stdev)
        stdev = K.sqrt(mean_sq_diff)
        # calculate the mean standard deviation across each pixel coord
        mean_pix = K.mean(stdev, keepdims=True)
        # scale this up to be the size of one input feature map for each sample
        shape = K.shape(inputs)
        output = K.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
        # concatenate with the output
        combined = K.concatenate([inputs, output], axis=-1)
        return combined

    def compute_output_signature(self, input_shape):
        # create a copy of the input shape as a list
        input_shape = list(input_shape)
        # add one to the channel dimension (assume channels-last)
        input_shape[-1] += 1
        # convert list to a tuple
        return tuple(input_shape)


class PixelNormalization(Layer):
    def __init__(self, **kwargs):
        """
        Provide activations statistical summary
        :param kwargs:
        """
        super(PixelNormalization, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # calculate square pixel values
        values = inputs ** 2.0
        # calculate the mean pixel values
        mean_values = K.mean(values, axis=-1, keepdims=True)
        # ensure the mean is not zero
        mean_values += 1.0e-8
        # calculate the sqrt of the mean squared value (L2 norm)
        l2 = K.sqrt(mean_values)
        # normalize values by the l2 norm
        normalized = inputs / l2
        return normalized

    def compute_output_signature(self, input_signature):
        return input_signature


class RandomWeightedAverage(merge._Merge):
    def __init__(self, batch_size):
        """
        Used for Gradient Penalty.
        Calculates interpolated images that lie at random points between the batch of real and fake images
        :param batch_size:
        """
        super().__init__()
        self.batch_size = batch_size

    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])
