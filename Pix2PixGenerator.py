from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import time
import threading
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from IPython.display import clear_output

BUFFER_SIZE = 100
BATCH_SIZE = 1
IMG_WIDTH = 64
IMG_HEIGHT = 64
LAMBDA = 100

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator():
    down_stack = [
        downsample(16, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(32, 4),  # (bs, 64, 64, 128)
        downsample(64, 4),  # (bs, 32, 32, 256)
        downsample(128, 4),  # (bs, 16, 16, 512)
        downsample(128, 4),  # (bs, 8, 8, 512)
        downsample(128, 4),  # (bs, 4, 4, 512)
    ]

    up_stack = [
        upsample(128, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(128, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(64, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(32, 4),  # (bs, 16, 16, 1024)
        upsample(16, 4),  # (bs, 32, 32, 512)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)

    last = tf.keras.layers.Conv2DTranspose(filters=3,
                                           kernel_size=4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(16, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(32, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(64, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(128, 3, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 3, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss

def initialize_pix2pix():
    generator = Generator()
    discriminator = Discriminator()
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_dir = './Datas/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    return generator

def generate_image(model, input):
    prediction = model(input, training=True)
    return prediction[0]

if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    pix2pix_generator = initialize_pix2pix()
    while True:
        if os.path.isfile("Datas/CreatedImageText/Image_0.csv"):
            print("Initiate Prediction by Generator")
            input_csv0 = pd.read_csv("Datas/CreatedImageText/Image_0.csv", header=None)
            input_csv0 = pd.DataFrame.to_numpy(input_csv0)
            input_csv1 = pd.read_csv("Datas/CreatedImageText/Image_1.csv", header=None)
            input_csv1 = pd.DataFrame.to_numpy(input_csv1)
            input_csv2 = pd.read_csv("Datas/CreatedImageText/Image_2.csv", header=None)
            input_csv2 = pd.DataFrame.to_numpy(input_csv2)

            arr = np.array([input_csv0, input_csv1, input_csv2])
            arr2 = np.zeros((64, 64, 3))
            for i in range(0, 64):
                for j in range(0, 64):
                    for k in range(0, 3):
                        arr2[i][j][k] = arr[k][i][j]
            map_data_tensor = tf.cast([arr2], tf.float32)

            pred = generate_image(pix2pix_generator, map_data_tensor)
            for i in range(2):
                os.remove(r"Datas/CreatedImageText/Image_"+str(i)+".csv")

            pred = pred.numpy()

            temp0 = [[0 for j in range(64)] for i in range(64)]
            temp1 = [[0 for j in range(64)] for i in range(64)]
            temp2 = [[0 for j in range(64)] for i in range(64)]

            for i in range(0, 64):
                for j in range(0, 64):
                    temp0[i][j] = pred[i][j][0]
                    temp1[i][j] = pred[i][j][1]
                    temp2[i][j] = pred[i][j][2]
            print("Writing prediction")
            f0 = open("Datas/CreatedImageData/prediction_type.csv", 'w', newline='')
            f1 = open("Datas/CreatedImageData/prediction_rot.csv", 'w', newline='')
            f2 = open("Datas/CreatedImageData/prediction_non.csv", 'w', newline='')
            wr0 = csv.writer(f0)
            wr1 = csv.writer(f1)
            wr2 = csv.writer(f2)
            for i in range(64):
                wr0.writerow(temp0[i])
                wr1.writerow(temp1[i])
                wr2.writerow(temp2[i])
            f0.close()
            f1.close()
            f2.close()
            print("Prediction is ended and Wait")
        time.sleep(20)
