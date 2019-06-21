import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def load_img_celeba(image_file, img_width, img_height, is_train):
    input_image = tf.read_file(image_file)
    input_image = tf.image.decode_jpeg(input_image, channels=3)

    target_img_file = tf.regex_replace(image_file, "landmarks", "photos")
    real_image = tf.read_file(target_img_file)
    real_image = tf.image.decode_jpeg(real_image)

    return _load_image(input_image, real_image, img_width, img_height, is_train)


def load_img_cuhk(image_file, img_width, img_height, is_train):
    input_image = tf.read_file(image_file)
    input_image = tf.image.decode_jpeg(input_image, channels=3)

    target_img_file = tf.regex_replace(image_file, "sketches", "photos")
    target_img_file = tf.regex_replace(target_img_file, "-sz1", "")
    target_img_file = tf.regex_replace(target_img_file, "F2-", "f-")
    target_img_file = tf.regex_replace(target_img_file, "M2-", "m-")
    real_image = tf.read_file(target_img_file)
    real_image = tf.image.decode_jpeg(real_image)

    return _load_image(input_image, real_image, img_width, img_height, is_train)


def load_image(image_file, img_width, img_height, is_train):
    image = tf.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    return _load_image(input_image, real_image, img_width, img_height, is_train)


def _load_image(input_image, real_image, img_width, img_height, is_train):
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    if is_train:
        # random jittering

        # resizing to IMG_WIDTH x IMG_HEIGHT x 3
        input_image = tf.image.resize_images(input_image, [img_width, img_height],
                                             align_corners=True,
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize_images(real_image, [img_width, img_height],
                                            align_corners=True,
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # randomly cropping to 256 x 256 x 3
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.random_crop(stacked_image, size=[2, img_height, img_width, 3])
        input_image, real_image = cropped_image[0], cropped_image[1]

        if np.random.random() > 0.5:
            # random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)
    else:
        input_image = tf.image.resize_images(input_image, size=[img_height, img_width],
                                             align_corners=True, method=2)
        real_image = tf.image.resize_images(real_image, size=[img_height, img_width],
                                            align_corners=True, method=2)

    # normalizing the images to [-1, 1]
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

class Downsample(tf.keras.Model):
    def __init__(self, filters, size, apply_batchnorm=True):
        super(Downsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Upsample(tf.keras.Model):
    def __init__(self, filters, size, apply_dropout=False):
        super(Upsample, self).__init__()
        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)

        self.up_conv = tf.keras.layers.Conv2DTranspose(filters,
                                                       (size, size),
                                                       strides=2,
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x1, x2, training):
        x = self.up_conv(x1)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.relu(x)
        x = tf.concat([x, x2], axis=-1)
        return x


class Generator(tf.keras.Model):
    def __init__(self, output_channels=3):
        super(Generator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = Downsample(64, 4, apply_batchnorm=False)
        self.down2 = Downsample(128, 4)
        self.down3 = Downsample(256, 4)
        self.down4 = Downsample(512, 4)
        self.down5 = Downsample(512, 4)
        self.down6 = Downsample(512, 4)
        self.down7 = Downsample(512, 4)
        self.down8 = Downsample(512, 4)

        self.up1 = Upsample(512, 4, apply_dropout=True)
        self.up2 = Upsample(512, 4, apply_dropout=True)
        self.up3 = Upsample(512, 4, apply_dropout=True)
        self.up4 = Upsample(512, 4)
        self.up5 = Upsample(256, 4)
        self.up6 = Upsample(128, 4)
        self.up7 = Upsample(64, 4)

        self.last = tf.keras.layers.Conv2DTranspose(output_channels,
                                                    (4, 4),
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer)

    @tf.contrib.eager.defun
    def call(self, x, training):
        # x shape == (bs, 256, 256, 3)
        x1 = self.down1(x, training=training)  # (bs, 128, 128, 64)
        x2 = self.down2(x1, training=training)  # (bs, 64, 64, 128)
        x3 = self.down3(x2, training=training)  # (bs, 32, 32, 256)
        x4 = self.down4(x3, training=training)  # (bs, 16, 16, 512)
        x5 = self.down5(x4, training=training)  # (bs, 8, 8, 512)
        x6 = self.down6(x5, training=training)  # (bs, 4, 4, 512)
        x7 = self.down7(x6, training=training)  # (bs, 2, 2, 512)
        x8 = self.down8(x7, training=training)  # (bs, 1, 1, 512)

        x9 = self.up1(x8, x7, training=training)  # (bs, 2, 2, 1024)
        x10 = self.up2(x9, x6, training=training)  # (bs, 4, 4, 1024)
        x11 = self.up3(x10, x5, training=training)  # (bs, 8, 8, 1024)
        x12 = self.up4(x11, x4, training=training)  # (bs, 16, 16, 1024)
        x13 = self.up5(x12, x3, training=training)  # (bs, 32, 32, 512)
        x14 = self.up6(x13, x2, training=training)  # (bs, 64, 64, 256)
        x15 = self.up7(x14, x1, training=training)  # (bs, 128, 128, 128)

        x16 = self.last(x15)  # (bs, 256, 256, 3)
        x16 = tf.nn.tanh(x16)

        return x16


class DiscDownsample(tf.keras.Model):
    def __init__(self, filters, size, apply_batchnorm=True):
        super(DiscDownsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = DiscDownsample(64, 4, False)
        self.down2 = DiscDownsample(128, 4)
        self.down3 = DiscDownsample(256, 4)

        # we are zero padding here with 1 because we need our shape to
        # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
        self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
        self.conv = tf.keras.layers.Conv2D(512,
                                           (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer,
                                           use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        # shape change from (batch_size, 31, 31, 512) to (batch_size, 30, 30, 1)
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        self.last = tf.keras.layers.Conv2D(1,
                                           (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer)

    @tf.contrib.eager.defun
    def call(self, inp, tar, training):
        # concatenating the input and the target
        x = tf.concat([inp, tar], axis=-1)  # (bs, 256, 256, channels*2)
        x = self.down1(x, training=training)  # (bs, 128, 128, 64)
        x = self.down2(x, training=training)  # (bs, 64, 64, 128)
        x = self.down3(x, training=training)  # (bs, 32, 32, 256)

        x = self.zero_pad1(x)  # (bs, 34, 34, 256)
        x = self.conv(x)  # (bs, 31, 31, 512)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.zero_pad2(x)  # (bs, 33, 33, 512)
        # don't add a sigmoid activation here since
        # the loss function expects raw logits.
        x = self.last(x)  # (bs, 30, 30, 1)

        return x


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(disc_real_output),
                                              logits = disc_real_output)
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.zeros_like(disc_generated_output),
                                                   logits = disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target, loss_lambda):
    gan_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(disc_generated_output),
                                               logits = disc_generated_output)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (loss_lambda * l1_loss)

    return total_gen_loss


def train_gan(generator, discriminator,
                generator_optimizer, discriminator_optimizer,
              input_image, target, loss_lambda):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator(input_image, target, training=True)
        disc_generated_output = discriminator(input_image, gen_output, training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target, loss_lambda=loss_lambda)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.variables)

    gen_train_step = generator_optimizer.apply_gradients(zip(generator_gradients, generator.variables))
    disc_train_step = discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                                  discriminator.variables))

    return gen_loss, disc_loss, gen_train_step, disc_train_step


def display_prediction(prediction, test_input, tar):
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
