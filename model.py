import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
    Concatenate, Conv2D, MaxPooling2D, \
    UpSampling2D, Reshape, GlobalAveragePooling2D, Dense, Layer, Flatten, Conv2DTranspose
from tensorflow import keras
import os
import sys

from data_gen import mnist_vae_loader
from visualizations import *


class Sampling(Layer):
    # def __int__(self,**kwargs):
    #     super(Sampling, self).__int__(**kwargs)
    def call(self,inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon

class Pretrain_block(Layer):
    def __init__(self,latent_dim=32,hidden_dim=64,name='encoder', is_conv=False, **kwargs):
        super(Encoder, self).__init__(name = name,**kwargs)
        self.dense_mean = Dense(latent_dim)

    def call(self,inputs):
        if self.is_conv:
            x = self.conv(inputs)
            x = self.Flatten(x)
            #x = Dense(16, activation="relu")(x)
        else:
            x = self.dense_hidden(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z_sample = self.sampling((z_mean,z_log_var))
        return z_mean,z_log_var,z_sample


class Unsampling_block(Layer):
    def __init__(self,latent_dim=32,hidden_dim=64,name='encoder', is_conv=False, **kwargs):
        super(Encoder, self).__init__(name = name,**kwargs)
        self.dense_hidden = Dense(hidden_dim,activation='relu')
        self.dense_mean = Dense(latent_dim)

    def call(self,inputs):
        x = self.conv(inputs)
        for i in range(4):
            x = self.conv(x)
        return x

class Encoder(Layer):
    def __init__(self,latent_dim=32,hidden_dim=64,name='encoder', is_conv=False, **kwargs):
        super(Encoder, self).__init__(name = name,**kwargs)
        self.dense_hidden = Dense(hidden_dim,activation='relu')
        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling()
        self.is_conv = is_conv
        self.Flatten = Flatten()
        self.conv = Conv2D(latent_dim, 3, activation="relu", strides=2, padding="same")

    def call(self,inputs):
        if self.is_conv:
            x = self.conv(inputs)
            x = self.Flatten(x)
            #x = Dense(16, activation="relu")(x)
        else:
            x = self.dense_hidden(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z_sample = self.sampling((z_mean,z_log_var))
        return z_mean,z_log_var,z_sample

class Decoder(Layer):
    def __init__(self,final_dim=32,hidden_dim=64,name='decoder',is_conv=False,**kwargs):
        super().__init__(name = name, **kwargs)
        self.dense_hidden = Dense(hidden_dim,activation='relu')
        self.dense_final = Dense(final_dim,activation='sigmoid')
        self.is_conv = is_conv
        self.convTranspose1 = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.convTranspose2 = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.convTranspose3 = Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")

        self.reshape = Reshape((7, 7, 32))
        self.reshape_small = Reshape((4, 4, 2))

        self.dense_to_meet_size = Dense(7 * 7 * 32, activation="relu")

        self.conv1D_final = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")

    def call(self,inputs):
        x = self.dense_hidden(inputs)

        if self.is_conv:
            # x = self.dense_to_meet_size(x)
            # x = self.reshape(x)
            x = self.reshape_small
            x = self.convTranspose3(x)
            x = self.convTranspose3(x)
            x = self.convTranspose3(x)


            x = self.convTranspose1(x)
            x = self.convTranspose2(x)
            z_final = self.conv1D_final(x)
        else:
            z_final = self.dense_final(x)
        return z_final

class VariationalAutoEncoder(Model):
    def __init__(self,original_dim,hidden_dim=64,latent_dim=32,name='auto_encoder',is_conv=False, **kwargs):
        super().__init__(name = name,**kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,hidden_dim=hidden_dim, is_conv=is_conv)
        self.decoder = Decoder(final_dim=original_dim,hidden_dim=hidden_dim, is_conv=is_conv)
        self.sampling = Sampling()

    def call(self,inputs):
        z_mean, z_log_var, z_sample = self.encoder(inputs)
        reconstructed = self.decoder(z_sample)
        kl_loss = -0.5 * tf.reduce_mean(z_log_var-tf.square(z_mean)-tf.exp(z_log_var)+1)
        self.add_loss(kl_loss)

        #mse_loss = tf.keras.losses.MeanSquaredError()(inputs, reconstructed)
        mse_loss = tf.keras.losses.BinaryCrossentropy()(inputs, reconstructed)

        self.add_loss(mse_loss)

        self.add_metric(kl_loss, name="kl_loss")
        self.add_metric(mse_loss, name="mse_loss")

        self.add_metric(self.losses, name="total_loss")

        return reconstructed
