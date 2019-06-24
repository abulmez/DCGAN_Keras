from keras import Sequential
from keras.optimizers import Adam


class DCGAN:
    def __init__(self, discriminator, generator):
        self.actual_dcgan = self.build_dcgan(discriminator, generator)

    def build_dcgan(self, discriminator, generator):
        model = Sequential()

        discriminator.trainable = False
        model.add(generator.actual_generator)
        model.add(discriminator.actual_discriminator)

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5),
                      metrics=['accuracy'])
        return model
