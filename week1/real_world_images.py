# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 2019/8/26 6:50 PM
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile

print(tf.__version__)
if __name__ == '__main__':
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('acc') > 0.99:
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    train_horse_dir = os.path.join('data/horse-or-human/horses')
    train_human_dir = os.path.join('data/horse-or-human/humans')
    train_horse_names = os.listdir(train_horse_dir)
    print(train_horse_names[:10])

    train_human_names = os.listdir(train_human_dir)
    print(train_human_names[:10])

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, kernel_size=3, activation="relu", input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.summary()

    # All images will be rescaled by 1./255
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

    train_generator = train_datagen.flow_from_directory(
        'data/horse-or-human/',
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary')

    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
                  metrics=['acc'])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        callbacks=[callbacks],
        verbose=1)

    print("hello, world")
