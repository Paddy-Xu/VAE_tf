import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
    Concatenate, Conv2D, MaxPooling2D, \
    UpSampling2D, Reshape, GlobalAveragePooling2D, Dense, Layer, Flatten, Conv2DTranspose
from tensorflow import keras
import os
import sys
from data_gen import *
tf.keras.backend.set_floatx('float64')


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs

        # eps = tf.random.normal(shape=z_mean.shape)
        # return eps * tf.exp(z_log_var * .5) + z_mean

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(Layer):
    def __init__(self, latent_dim=32, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling()
        self.Flatten = Flatten()
        self.conv32 = Conv2D(32, 3, activation="relu", strides=2, padding="same")
        self.conv64 = Conv2D(64, 3, activation="relu", strides=2, padding="same")

    def call(self, inputs):
        x = self.conv32(inputs)
        x = self.conv64(inputs)

        x = self.Flatten(x)

        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z_sample = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z_sample


class Decoder(Layer):
    def __init__(self, name='decoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.convTranspose32 = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.convTranspose64 = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.reshape = Reshape((7, 7, 32))

        self.dense_to_meet_size = Dense(7 * 7 * 32, activation="relu")

        self.conv1D_final = Conv2DTranspose(1, 3,
                                            # activation="sigmoid",
                                            padding="same")

    def call(self, inputs):
        x = self.dense_to_meet_size(inputs)
        x = self.reshape(x)
        x = self.convTranspose64(x)
        x = self.convTranspose32(x)
        z_final = self.conv1D_final(x)

        return z_final


class VAE_w_grad(Model):

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)

        log2pi = tf.cast(log2pi, tf.float64)
        sample = tf.cast(sample, tf.float64)
        mean = tf.cast(mean, tf.float64)
        logvar = tf.cast(logvar, tf.float64)

        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def __init__(self, original_dim, latent_dim=32, name='auto_encoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder()
        self.sampling = Sampling()

    #@tf.function
    def call(self, inputs):  ##TODO: get a tf decrator?
        #print(inputs.shape)
        x, gradient = inputs

        results = []
        for name, inputs in zip(['image', 'gradient'], [x, gradient]):
            z_mean, z_log_var, z_sample = self.encoder(inputs)
            reconstructed = self.decoder(z_sample)
            kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstructed,
                                                                labels=inputs)

            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
            logpz = self.log_normal_pdf(z_sample, 0., 0.)
            logqz_x = self.log_normal_pdf(z_sample, z_mean, z_log_var)

            # self.add_loss(-tf.reduce_mean(logpx_z + logpz - logqz_x))

            if name == 'gradient':
                mse_loss = tf.keras.losses.MeanSquaredError()(inputs, reconstructed)
                self.add_loss(mse_loss)
                self.add_metric(mse_loss, name=f"{name}_mse_loss")
            else:
                ce_loss = tf.reduce_mean(cross_ent)
                self.add_loss(ce_loss)
                self.add_metric(ce_loss, name=f"{name}_cross_loss")

            kl_loss *= 0.05
            self.add_loss(kl_loss)
            self.add_metric(kl_loss, name=f"{name}_kl_loss")

        # self.add_metric(mse_loss, name="mse_loss")

        # self.add_metric(self.losses, name="total_loss")
        results.append(reconstructed)
        return results


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 128
vae = VAE_w_grad(original_dim=784, latent_dim=2)

optimizer = tf.keras.optimizers.Adam(1e-4)

vae.compile(optimizer)

data_gen = mnist_vae_loader_w_gradient(batch_size=batch_size)
train_dataset, test_dataset = data_gen.get_dataset()
#train_dataset = train_dataset.repeat()
(img, gradient), y = iter(train_dataset).next()

row,col = 3, 3
fig, axs = plt.subplots(nrows=row, ncols=col, constrained_layout=False)

for r in range(row):
    for c in range(col):
        axs[r, c].imshow(tf.reshape(gradient[r + c], shape=(28, 28)).numpy().astype("float32"),
                         cmap='gray')
plt.show()

vae.fit(train_dataset, epochs=15)

#vae.fit(iter(train_dataset).next(), epochs=15, steps_per_epoch=2000)
