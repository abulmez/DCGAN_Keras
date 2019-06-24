from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, GaussianNoise, Flatten, Dense
from keras.optimizers import Adam


class Discriminator:

    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.actual_discriminator = self.build_discriminator()

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(16, kernel_size=4, strides=2,
                         input_shape=self.img_shape, padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(GaussianNoise(0.1))

        model.add(Conv2D(32, kernel_size=4, strides=2,
                         padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(GaussianNoise(0.1))

        model.add(Conv2D(64, kernel_size=4, strides=2,
                         padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(GaussianNoise(0.1))

        model.add(Conv2D(128, kernel_size=4, strides=2,
                         padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(GaussianNoise(0.1))

        model.add(Conv2D(256, kernel_size=4, strides=2,
                         padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(GaussianNoise(0.1))

        model.add(Conv2D(512, kernel_size=4, strides=2,
                         padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(GaussianNoise(0.1))

        model.add(Conv2D(1024, kernel_size=4, strides=2,
                         padding='same'))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
        return model

    def load_discriminator_weights(self, saved_weights_file_name):
        self.actual_discriminator.load_weights("./../model_backup/" + saved_weights_file_name)

    def save_discriminator_weights(self, current_time, iteration):
        self.actual_discriminator.save_weights(
            "./../model_backup/" + current_time + "ep" + str(iteration) + "discriminator.h5")
