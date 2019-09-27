import os
import numpy as np
from tensorflow.python.keras.layers import Dense, Flatten, Input, Reshape
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.layers import LeakyReLU, BatchNormalization, Lambda, Activation
import tensorflow as tf
from tqdm import tqdm

from gan_utils import conv, get_initial_size, PlotData, upscale

# TODO
# * best params for Conv2DTranspose
# * add pixelshuffler
# * different kernel size for last gen layer??


class DCGan:
    def __init__(self, input_shape, config):
        self.config = config
        self.config['model']['generator']['z_size'] = self.config['data']['z_size']
        self.input_shape = input_shape

        self._build_model()

    def _build_model(self):
        # Generator
        model_input = Input(self.config['data']['z_size'], name='gan_input')
        self.generator = DCGan._build_generator(model_input, self.input_shape, self.config['model']['generator'])

        # Discriminator
        self.discriminator = DCGan._build_discriminator(self.input_shape, self.config['model']['discriminator'])

        # GAN
        #self.gan = Sequential([self.generator, self.discriminator])
        model_output = self.discriminator(self.generator(model_input))
        self.gan = Model(model_input, model_output)

        # Compile discriminator
        # discriminator_optimizer = RMSprop(lr=self.config['training']['discriminator']['learning_rate'])
        # self.discriminator.compile(loss="binary_crossentropy", optimizer=discriminator_optimizer,
        #                            metrics=['accuracy'])
        #
        # # Compile generator
        # # taken into account only when compiling a model,
        # # so discriminator is trainable if we call its fit() method while not for the gan model
        # self.discriminator.trainable = False
        #
        # gan_optimizer = RMSprop(lr=self.config['training']['generator']['learning_rate'])
        # self.gan.compile(loss="binary_crossentropy", optimizer=gan_optimizer,
        #                  metrics=['accuracy'])
        #
        # self.discriminator.trainable = True

    # Already with a basic GAN setup we break the use of model.fit and related utilities
    # need to find a way to manage callbacks and validation

    # This still doesn't work for a problem possibly related to a bug with nested models https://github.com/keras-team/keras/issues/10074
    # Not compiling the discriminator in fact doesn't trigger the error anymore
    def train(self, train_ds, validation_ds, nb_epochs: int, log_dir, checkpoint_dir, is_tfdataset=False):
        callbacks = []

        # tensorboard
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)

        # checkpoints
        if checkpoint_dir:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                             save_weights_only=True,
                                                             verbose=1,
                                                             period=self.config['training']['checkpoint_steps'])
            callbacks.append(cp_callback)

        # plotting callback
        plot_callback = PlotData(validation_ds, self.generator, log_dir)
        callbacks.append(plot_callback)

        # training
        batch_size = self.config['training']['batch_size']
        z_dim = self.config['data']['z_size']
        for epoch in range(nb_epochs):
            if is_tfdataset:
                for x in train_ds:
                    train_batch = x.numpy()
                    break
            else:
                idx = np.random.randint(0, train_ds.shape[0], batch_size)
                train_batch = train_ds[idx]

            self.train_discriminator(train_batch, batch_size, z_dim)
            self.train_generator(batch_size, z_dim)

            # TODO add validation step

    # Train with pure TF, because Keras doesn't work
    def _train(self, train_ds, validation_ds, nb_epochs: int, log_dir, checkpoint_dir, is_tfdataset=False,
               restore_latest_checkpoint=True):
        batch_size = self.config['training']['batch_size']
        z_dim = self.config['data']['z_size']

        noise = tf.random.normal([batch_size, z_dim])
        plot_summary_writer = tf.summary.create_file_writer(str(log_dir / 'plot'))
        train_summary_writer = tf.summary.create_file_writer(str(log_dir / 'train'))

        # optimizers
        generator_optimizer = tf.keras.optimizers.Adam(self.config['training']['generator']['learning_rate'])
        discriminator_optimizer = tf.keras.optimizers.Adam(self.config['training']['discriminator']['learning_rate'])

        # checkpoints
        if checkpoint_dir:
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                             discriminator_optimizer=discriminator_optimizer,
                                             generator=self.generator,
                                             discriminator=self.discriminator)
            ckpt_manager = tf.train.CheckpointManager(checkpoint, os.path.join(checkpoint_dir, "ckpt"),
                                                      max_to_keep=self.config['training']['checkpoints_to_keep'])
            if restore_latest_checkpoint and ckpt_manager.latest_checkpoint:
                print(f"Restored from {ckpt_manager.latest_checkpoint}")
            else:
                print("Initializing from scratch.")

        # train loop
        for epoch in tqdm(range(nb_epochs)):
            gen_losses = []
            disc_losses = []
            for ds_batch in train_ds:
                gen_loss, disc_loss = train_step(ds_batch, self.generator, self.discriminator,
                                                 generator_optimizer=generator_optimizer,
                                                 discriminator_optimizer=discriminator_optimizer,
                                                 batch_size=batch_size, noise_dim=z_dim)
                gen_losses.append(gen_loss)
                disc_losses.append(disc_loss)

            # Loss summary
            avg_gen_loss = np.mean(gen_losses)
            avg_disc_loss = np.mean(disc_losses)
            with train_summary_writer.as_default():
                tf.summary.scalar("Average Gen Loss", avg_gen_loss, step=epoch)
                tf.summary.scalar("Average Disc Loss", avg_disc_loss, step=epoch)

            # Plot data
            with plot_summary_writer.as_default():
                # Plot sample data
                predictions = self.generator(noise)
                tf.summary.image("Sample Generated", predictions, step=epoch)
                tf.summary.image("Sample Input", [ds_batch[np.random.randint(len(ds_batch))]], step=epoch)

            # checkpoint
            if checkpoint_dir:
                checkpoint.step.assign_add(1)
                ckpt_step = int(checkpoint.step)
                if ckpt_step % self.config['training']['checkpoint_steps'] == 0:
                    save_path = ckpt_manager.save()
                    print(f"Saved checkpoint for step {ckpt_step}: {save_path}")

    @staticmethod
    # takes an image and generates two vectors: means and standards deviations
    def _build_generator(model_input, img_shape, config):
        latent_vector = Input(config['z_size'], name='generator_input')
        init_shape = tuple([get_initial_size(d, config['num_conv_blocks'])
                            for d in img_shape[:-1]] + [config['init_filters']])

        x = Dense(np.prod(init_shape))(latent_vector)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Reshape(init_shape)(x)

        for i in range(config['num_conv_blocks'] - 1):
            x = upscale(filters=config['init_filters'] // 2 ** i,
                        kernel_size=config['kernel_size'], strides=config['strides'],
                        upscale_method=config['upscale_method'],
                        activation='relu')(x)

        # last upscale layer
        model_output = upscale(filters=config['n_channels'],
                               kernel_size=config['kernel_size'], strides=config['strides'],
                               upscale_method=config['upscale_method'],
                               activation='tanh')(x)

        return Model(latent_vector, model_output)

    @staticmethod
    def _build_discriminator(img_shape, config):
        model_input = Input(shape=img_shape, name="discriminator_input")

        x = model_input
        for i in range(config['num_conv_blocks']):
            x = conv(filters=config['init_filters'] * (2 ** i), kernel_size=config['kernel_size'],
                     strides=config['strides'])(x)

        features = Flatten()(x)

        model_output = Dense(1, activation='sigmoid')(features)

        return Model(model_input, model_output)

    def train_discriminator(self, true_imgs, batch_size: int, z_dim: int):
        # Train on real image
        # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
        self.discriminator.train_on_batch(true_imgs, np.ones((batch_size, 1)))

        # Train on generated images
        # [0,0,...,0] with generated images since they are fake
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = self.generator.predict(noise)
        self.discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))

    def train_generator(self, batch_size: int, z_dim: int):
        # Train on noise input
        # [1,1,...,1] with generated output since we want the discriminator to believe these are real images
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        self.gan.train_on_batch(noise, np.ones((batch_size, 1)))

    def setup_dataset(self, dataset):
        # prefetch lets the dataset fetch batches in the background while the model is training
        dataset = dataset.shuffle(self.config['data']['buffer_size']) \
                            .batch(self.config['training']['batch_size'], drop_remainder=True) \
                            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset


# Add second parameter for Keras compatibility
def generator_loss(generated_output, _=None):
    return tf.keras.losses.binary_crossentropy(tf.ones_like(generated_output), generated_output)


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(generated_output), generated_output)

    total_loss = real_loss + generated_loss

    return total_loss


@tf.function
def train_step(images, generator, discriminator,
               generator_optimizer, discriminator_optimizer,
               batch_size, noise_dim):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # generating noise from a normal distribution
        noise = tf.random.normal([batch_size, noise_dim])

        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_train_step = generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_train_step = discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss
