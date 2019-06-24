import os
import random
import time
from tkinter import END

import numpy
from PIL import Image, ImageTk

from model.DCGAN import DCGAN
from model.Discriminator import Discriminator
from model.Generator import Generator

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

plotly.tools.set_credentials_file(username='bair2059', api_key='R8rbIch9IbzPmFtrkbBM')


class DCGANTrainer:

    def __init__(self, training_epochs, batch_size, training_data_folder_name, z_dim, sampling_interval,
                 target_img_height,
                 target_img_width,
                 canvas, console):
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.training_data_folder_name = training_data_folder_name
        self.z_dim = z_dim
        self.sampling_interval = sampling_interval
        self.target_img_height = target_img_height
        self.target_img_width = target_img_width
        self.img_names_list = None
        self.img_shape = None
        self.load_img_names_list()
        self.determine_img_shape()
        self.generator = Generator(self.z_dim, self.img_shape)
        self.discriminator = Discriminator(self.img_shape)
        self.load_saved_weights()
        self.dcgan = DCGAN(self.discriminator, self.generator)
        self.training_images = []
        self.load_training_images()
        self.random_noise_for_sampling = None
        self.canvas = canvas
        self.canvas_image = None
        self.current_displayed_image = None
        self.console = console

    def load_img_names_list(self):
        self.img_names_list = os.listdir("./../data/" + self.training_data_folder_name)

    def determine_img_shape(self):
        pic = Image.open("./../data/" + self.training_data_folder_name + "/" + self.img_names_list[0])
        channels = len(pic.getbands())
        self.img_shape = (self.target_img_height, self.target_img_width, channels)

    def load_saved_weights(self):
        saved_weights_list = os.listdir("./../model_backup")
        if len(saved_weights_list) >= 2:
            self.generator.load_generator_weights(saved_weights_list[-1])
            self.discriminator.load_discriminator_weights(saved_weights_list[-2])

    def pillow_image_to_normalized_numpy_array(self, pillow_image):
        pix = numpy.array(pillow_image)
        pix = pix / 127.5 - 1.
        return pix

    def load_training_images(self):
        for img_name in self.img_names_list:
            pic = Image.open("./../data/" + self.training_data_folder_name + "/" + img_name)
            pic = pic.resize((self.target_img_height, self.target_img_width), resample=Image.LANCZOS)
            self.training_images.append(pic)

    def sample_images(self, iteration):

        grid_width_size = 5
        grid_height_size = 5
        if self.random_noise_for_sampling is None:
            self.random_noise_for_sampling = np.random.normal(0, 1,
                                                              (grid_width_size * grid_height_size, self.z_dim))

        # Generate images from random noise
        gen_imgs = self.generator.actual_generator.predict(self.random_noise_for_sampling)

        # Rescale images to 0-1
        gen_imgs = 127.5 * (gen_imgs + 1)
        gen_imgs = gen_imgs.astype(int)
        grid = Image.new('RGB', (grid_width_size * self.target_img_width, grid_width_size * self.target_img_height),
                         (255, 255, 255))
        for i in range(0, grid_width_size):
            for j in range(0, grid_height_size):
                sample = Image.fromarray(numpy.uint8(gen_imgs[grid_width_size * i + j]))
                grid.paste(sample, (i * self.target_img_width, j * self.target_img_height))
        grid.save("./../samples/" + str(int(time.time())) + "ep" + str(iteration) + ".jpg", "JPEG")
        self.current_displayed_image = ImageTk.PhotoImage(grid)
        if self.canvas_image is None:
            self.canvas_image = self.canvas.create_image(325, 325, image=self.current_displayed_image)
        else:
            self.canvas.itemconfigure(self.canvas_image, image=self.current_displayed_image)

    def train(self):
        self.console.insert(END, "Starting training process")

        discriminator_losses = []
        generator_losses = []
        discriminator_accuracies = []
        batch_iterations_per_epoch = int(len(self.training_images) // self.batch_size)
        counter = -1
        overall_counter = 0

        for ep in range(self.training_epochs):

            random.shuffle(self.training_images)

            for it in range(batch_iterations_per_epoch):

                counter += 1

                # -------------------------
                #  Train the Discriminator
                # -------------------------

                # Select a random batch of real images

                real = (np.ones(self.batch_size) -
                        np.random.random_sample(self.batch_size) * 0.1)
                fake = np.random.random_sample(self.batch_size) * 0.1

                converted_imgs = []
                for i in range(it * self.batch_size, it * self.batch_size + self.batch_size):
                    converted_imgs.append(self.pillow_image_to_normalized_numpy_array(self.training_images[i]))
                converted_imgs = numpy.array(converted_imgs)

                # Generate a batch of fake images
                z = np.random.normal(0, 1, (self.batch_size, self.z_dim))
                gen_imgs = self.generator.actual_generator.predict(z)

                # Discriminator loss

                # if last_d_loss > 0.100000:
                self.discriminator.actual_discriminator.trainable = True

                d_loss_real = self.discriminator.actual_discriminator.train_on_batch(converted_imgs, real)
                d_loss_fake = self.discriminator.actual_discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                self.discriminator.actual_discriminator.trainable = False

                # ---------------------
                #  Train the Generator
                # ---------------------

                # Generate a batch of fake images
                z = np.random.normal(0, 1, (self.batch_size, self.z_dim))

                # Generator loss
                g_loss = self.dcgan.actual_dcgan.train_on_batch(z, real)

                if counter % self.sampling_interval == 0:
                    # Output training progress
                    console_output = "Ep. {} It. {} [D loss: {:f}] [G loss: {:f}]".format(ep, it, d_loss[0],
                                                                                          g_loss[0])
                    self.console.insert(END, "\n" + console_output)

                    # Save losses and accuracies so they can be plotted after training
                    overall_counter += 1
                    discriminator_losses.append(d_loss[0])
                    generator_losses.append(g_loss[0])
                    discriminator_accuracies.append(100 * d_loss[1])

                    # Output generated image samples
                    self.sample_images(ep)

                if counter % 1000 == 0:
                    counter = 0
                    self.save_model_weights(ep)

        it = list(range(1, overall_counter))

        layout = go.Layout(
            yaxis=dict(
                range=[0, 5]
            )
        )

        discriminator_trace = go.Scatter(
            x=it,
            y=discriminator_losses,
            mode='lines',
            name='Discriminator loss'
        )

        generator_trace = go.Scatter(
            x=it,
            y=generator_losses,
            mode='lines',
            name='Generator loss'
        )
        discriminator_fig = go.Figure(data=[discriminator_trace], layout=layout)
        generator_fig = go.Figure(data=[generator_trace], layout=layout)
        py.plot(discriminator_fig, filename='discriminator_losses')
        py.plot(generator_fig, filename='generator_losses')

    def save_model_weights(self, iteration):
        self.console.insert(END, "\n" + "Backing_up weights")
        current_time = str(int(time.time()))
        self.generator.save_generator_weights(current_time, iteration)
        self.discriminator.save_discriminator_weights(current_time, iteration)

    def run_tests(self):
        z = np.random.normal(0, 1, (1000, self.z_dim))
        fake_images = self.generator.actual_generator.predict(z)
        fake_prediction_vector = self.discriminator.actual_discriminator.predict(fake_images)
        random.shuffle(self.training_images)
        converted_imgs = []
        for i in range(0, 1000):
            converted_imgs.append(self.pillow_image_to_normalized_numpy_array(self.training_images[i]))
        converted_imgs = numpy.array(converted_imgs)
        real_prediction_vector = self.discriminator.actual_discriminator.predict(converted_imgs)

        correct = 0

        for i in range(0, len(fake_prediction_vector)):
            if fake_prediction_vector[i] < 0.5:
                correct += 1
            if real_prediction_vector[i] >= 0.5:
                correct += 1

        self.console.insert(END, "\n" + "Discriminator accuracy: " + str(100 * (correct / 2000)))
