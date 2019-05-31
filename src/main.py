import gc
import os
import random
import tensorflow
import time
from builtins import range

import numpy
from keras.activations import selu, relu
from keras.backend import get_session, clear_session, set_session
from keras.layers import Input, Dense, Reshape, Flatten, GaussianNoise, UpSampling2D
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from PIL import Image

import numpy as np
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer("image_width", 28, "")
flags.DEFINE_integer("image_height", 28, "")
flags.DEFINE_string("training_data_folder_name", "mnist", "")
flags.DEFINE_integer("batch_size", 128, "")
flags.DEFINE_integer("epochs", 1000, "")

img_height = None
img_width = None
channels = None
training_data_folder_name = None
img_shape = None
img_names_list = []

z_dim = 2048

generator = None
discriminator = None
combined_dcgan = None
z = Input(shape=(z_dim,))
img = None
losses = []
accuracies = []
training_images = []


def build_generator(z_dim):
    model = Sequential()
    # Reshape input into 8x8x256 tensor via a fully connected layer
    model.add(Dense(2048 * 1 * 1, input_shape=(z_dim,)))
    # model.add(Activation('selu'))
    # model.add(BatchNormalization(momentum=0.5))
    # Leaky ReLU

    model.add(Reshape((1, 1, 2048)))
    model.add(Conv2DTranspose(
        2048, kernel_size=4))

    # # Transposed convolution layer, from 8x8x256 into 16x16x128 tensor
    # model.add(Conv2DTranspose(
    #     128, kernel_size=3, strides=2, padding='same'))
    # # model.add(Conv2DTranspose(
    # #     128, kernel_size=3, strides=2, padding='same'))
    #
    # # Batch normalization
    # model.add(BatchNormalization(momentum=0.5))
    # # Leaky ReLU
    model.add(Activation('relu'))

    # Transposed convolution layer, from 16x16x128 to 32x32x64 tensor
    model.add(Conv2DTranspose(
        1024, kernel_size=5, strides=2, padding='same'))

    # Batch normalization
    model.add(BatchNormalization(momentum=0.5))

    # Leaky ReLU
    # model.add(LeakyReLU(alpha=0.2))
    model.add(Activation('relu'))
    # model.add(UpSampling2D())
    # model.add(Activation('selu'))

    # Transposed convolution layer, from 32x32x64 to 64x64x32 tensor
    model.add(Conv2DTranspose(
        512, kernel_size=5, strides=2, padding='same'))
    # Batch normalization
    model.add(BatchNormalization(momentum=0.5))
    # Leaky ReLU
    model.add(Activation('relu'))
    # model.add(UpSampling2D())
    # model.add(Activation('selu'))
    # Transposed convolution layer, from 32x32x64 to 64x64x32 tensor

    model.add(Conv2DTranspose(
        256, kernel_size=5, strides=2, padding='same'))
    # Batch normalization
    model.add(BatchNormalization(momentum=0.5))
    # Leaky ReLU
    # model.add(Activation('selu'))
    model.add(Activation('relu'))
    # model.add(UpSampling2D())
    model.add(Conv2DTranspose(
        128, kernel_size=5, strides=2, padding='same'))
    # Batch normalization
    model.add(BatchNormalization(momentum=0.5))
    # Leaky ReLU
    # model.add(Activation('selu'))
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(
        64, kernel_size=5, strides=2, padding='same'))
    # Batch normalization
    model.add(BatchNormalization(momentum=0.5))
    # Leaky ReLU
    # model.add(Activation('selu'))
    model.add(Activation('relu'))
    # model.add(UpSampling2D())
    # Transposed convolution layer, from 64x64x32 to 128x128x3 tensor
    model.add(Conv2DTranspose(
        channels, kernel_size=5, strides=1, padding='same'))
    # Tanh activation
    model.add(Activation('tanh'))

    return model


def build_discriminator(img_shape):
    model = Sequential()

    # Convolutional layer, from 128x128x3 into 64x64x64 tensor
    model.add(Conv2D(16, kernel_size=4, strides=2,
                     input_shape=img_shape, padding='same'))
    # Leaky ReLU
    model.add(GaussianNoise(0.1))
    model.add(LeakyReLU(alpha=0.2))
    # Convolutional layer, from 64x64x64  into 32x32x64 tensor
    model.add(Conv2D(32, kernel_size=4, strides=2,
                     padding='same'))
    # Batch normalization
    model.add(BatchNormalization(momentum=0.5))
    # Leaky ReLU
    model.add(GaussianNoise(0.1))
    model.add(LeakyReLU(alpha=0.2))
    # Convolutional layer, from 32x32x64 tensor into 16x16x64 tensor
    model.add(Conv2D(64, kernel_size=4, strides=2,
                     padding='same'))
    # Batch normalization
    model.add(BatchNormalization(momentum=0.5))
    # Leaky ReLU
    model.add(GaussianNoise(0.1))
    model.add(LeakyReLU(alpha=0.2))
    # Convolutional layer, from 16x16x64 tensor into 8x8x64 tensor
    model.add(Conv2D(128, kernel_size=4, strides=2,
                     padding='same'))
    # Batch normalization
    model.add(BatchNormalization(momentum=0.5))
    # Leaky ReLU
    model.add(GaussianNoise(0.1))
    model.add(LeakyReLU(alpha=0.2))
    # Convolutional layer, from 8x8x64 tensor into 4x4x64 tensor
    model.add(Conv2D(256, kernel_size=4, strides=2,
                     padding='same'))

    model.add(GaussianNoise(0.1))
    model.add(LeakyReLU(alpha=0.2))
    # Convolutional layer, from 8x8x64 tensor into 4x4x64 tensor
    model.add(Conv2D(512, kernel_size=4, strides=2,
                     padding='same'))

    model.add(GaussianNoise(0.1))
    model.add(LeakyReLU(alpha=0.2))
    # Convolutional layer, from 8x8x64 tensor into 4x4x64 tensor
    model.add(Conv2D(1024, kernel_size=4, strides=2,
                     padding='same'))
    # Batch normalization
    # model.add(BatchNormalization(momentum=0.5))
    # # Leaky ReLU
    # model.add(Activation('selu'))
    # model.add(Conv2D(256, kernel_size=4, strides=1,
    #                  padding='same'))
    # Batch normalization
    # Flatten the tensor and apply sigmoid activation function
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


def build_combined_dcgan():
    global combined_dcgan
    combined_dcgan = Sequential()
    # Only false for the adversarial model
    discriminator.trainable = False
    combined_dcgan.add(generator)
    combined_dcgan.add(discriminator)

    combined_dcgan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5),
                           metrics=['accuracy'])


def build_networks():
    global discriminator, generator, z, img
    # Build and compile the Discriminator
    discriminator = build_discriminator(img_shape)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

    # Build the Generator
    generator = build_generator(z_dim)
    generator.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

    load_saved_weights()
    build_combined_dcgan()


def load_training_images():
    global training_images
    for img_name in img_names_list:
        pic = Image.open("./../data/" + training_data_folder_name + "/" + img_name)
        pic = pic.resize((img_height, img_width), resample=Image.LANCZOS)
        training_images.append(pic)
    # training_images = numpy.array(training_images)


def pillow_images_to_normalized_rgb(pillow_image):
    pix = numpy.array(pillow_image)
    pix = pix / 127.5 - 1.
    return pix


count = -1


def train(iterations, batch_size, sample_interval):
    global count
    # Load the dataset
    load_training_images()
    # Labels for real and fake examples
    # real = np.ones(batch_size)
    # fake = np.zeros(batch_size)

    # idx = np.random.randint(0, pillow_images_to_normalized_rgb(training_images[0]).shape[0], 1000)
    #
    # converted_imgs = []
    # for i in range(0, len(idx)):
    #     converted_imgs.append(pillow_images_to_normalized_rgb(training_images[idx[i]]))
    # converted_imgs = numpy.array(converted_imgs)
    #
    # real = (np.ones(1000) -
    #         np.random.random_sample(1000) * 0.1)
    #
    # discriminator.trainable = True
    # discriminator.train_on_batch(converted_imgs, real)

    for iteration in range(iterations):

        # -------------------------
        #  Train the Discriminator
        # -------------------------

        # Select a random batch of real images

        real = (np.ones(batch_size) -
                np.random.random_sample(batch_size) * 0.1)
        fake = np.random.random_sample(batch_size) * 0.1

        idx = np.random.randint(0, pillow_images_to_normalized_rgb(training_images[0]).shape[0], batch_size)

        converted_imgs = []
        for i in range(0, len(idx)):
            converted_imgs.append(pillow_images_to_normalized_rgb(training_images[idx[i]]))
        converted_imgs = numpy.array(converted_imgs)

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        # Discriminator loss

        # if last_d_loss > 0.100000:
        discriminator.trainable = True

        actual_real = real
        actual_fake = fake

        count += 1

        if count == 10:
            actual_real = fake
            actual_fake = real
            count = -1

        d_loss_real = discriminator.train_on_batch(converted_imgs, actual_real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, actual_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        discriminator.trainable = False

        # ---------------------
        #  Train the Generator
        # ---------------------

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, z_dim))

        # Generator loss
        g_loss = combined_dcgan.train_on_batch(z, real)
        # g_loss = combined_dcgan.train_on_batch(z, real)

        if iteration % sample_interval == 0:
            # Output training progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (iteration, d_loss[0], 100 * d_loss[1], g_loss[0]))

            # Save losses and accuracies so they can be plotted after training
            losses.append((d_loss[0], g_loss[0]))
            accuracies.append(100 * d_loss[1])

            # Output generated image samples
            sample_images(iteration)

        if iteration % 1000 == 0:
            save_model_weights(iteration)


def save_model_weights(iteration):
    print("Backing_up weights")
    current_time = str(int(time.time()))
    discriminator.save_weights("./../model_backup/" + current_time + "ep" + str(iteration) + "discriminator.h5")
    generator.save_weights("./../model_backup/" + current_time + "ep" + str(iteration) + "generator.h5")


def load_saved_weights():
    saved_weights_list = os.listdir("./../model_backup")
    if len(saved_weights_list) >= 2:
        generator.load_weights("./../model_backup/" + saved_weights_list[-1])
        discriminator.load_weights("./../model_backup/" + saved_weights_list[-2])


random_noise = None


def sample_images(iteration):
    global random_noise
    # Sample random noise
    grid_width_size = 5
    grid_height_size = 5
    if random_noise is None:
        random_noise = np.random.normal(0, 1,
                                        (grid_width_size * grid_height_size, z_dim))

    # Generate images from random noise
    gen_imgs = generator.predict(random_noise)

    # Rescale images to 0-1
    gen_imgs = 128 * (gen_imgs + 1)
    gen_imgs = gen_imgs.astype(int)
    grid = Image.new('RGB', (grid_width_size * img_width, grid_width_size * img_height), (255, 255, 255))
    for i in range(0, grid_width_size):
        for j in range(0, grid_height_size):
            sample = Image.fromarray(numpy.uint8(gen_imgs[grid_width_size * i + j]))
            grid.paste(sample, (i * img_width, j * img_height))
    grid.save("./../samples/" + str(int(time.time())) + "ep" + str(iteration) + ".jpg", "JPEG")


epochs = 20000
batch_size = 128
sample_interval = 100


def init_values_from_flags():
    global batch_size, epochs, training_data_folder_name
    batch_size = flags.FLAGS.batch_size
    epochs = flags.FLAGS.epochs
    training_data_folder_name = flags.FLAGS.training_data_folder_name


def get_nearest_multiple_of_2(image_height, image_width):
    min_dimension = min(image_height, image_width)
    power_of_2 = 1
    while power_of_2 < min_dimension:
        power_of_2 = power_of_2 * 2
    return power_of_2 / 2


def init_other_global_variables():
    global img_shape, img_names_list, channels, img_width, img_height
    img_names_list = os.listdir("./../data/" + training_data_folder_name)
    pic = Image.open("./../data/" + training_data_folder_name + "/" + img_names_list[0])
    channels = len(pic.getbands())
    img_width = img_height = int(get_nearest_multiple_of_2(pic.height, pic.width))
    img_shape = (img_height, img_width, channels)


# Train the GAN for the specified number of iterations


def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()
    print(gc.collect())  # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))


def main():
    # reset_keras()
    init_values_from_flags()
    init_other_global_variables()
    build_networks()
    train(epochs, batch_size, sample_interval)


main()
