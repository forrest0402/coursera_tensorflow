# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 2019/8/24 4:50 PM
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)
if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(len(x_train))
    print(len(x_test))
    print("train_data shape:", x_train.shape, "train_label shape:", y_train.shape)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    print("train_data shape:", x_train.shape, "test_data shape:", x_test.shape)

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('acc') > 0.99:
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True


    callbacks = myCallback()

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(kernel_size=2, filters=64, padding="same", activation=tf.nn.relu,
                               input_shape=(28, 28, 1)),
        # tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=6, callbacks=[callbacks])
    print("evaluate")
    print(model.evaluate(x_test, y_test))
    print("hello")
