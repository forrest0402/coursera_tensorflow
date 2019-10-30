# -*- coding: utf-8 -*-

"""
@Author: xiezizhe 
@Date: 2019/10/30 5:03 PM
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

print(tf.__version__)


class InfersentLayer(tf.keras.layers.Layer):

    def __init__(self, filters, filter_depth, emb_dim):
        super(InfersentLayer, self).__init__()


    def call(self, input_tensor):
        branches = []


class Infersent(tf.keras.Model):

    def __init__(self, vocab_size, max_len, emb_dim, filter_depth, output_size,
                 filters=[2, 2, 3, 3, 5, 5, 7, 7],
                 rate=0.5,
                 task_output=tf.keras.layers.Dense(1, activation='sigmoid')):
        super(Infersent, self).__init__()

        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, emb_dim, input_length=max_len),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, activation="relu")),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Reshape((max_len, emb_dim, 1)),
            InfersentLayer(filters, filter_depth, emb_dim),
            tf.keras.layers.Dense(output_size),
            tf.keras.layers.Dropout(rate),
            task_output
        ])
        self.model.summary()

    def call(self, x, y, training):
        return self.model(inputs, training=training)


if __name__ == '__main__':
    pass
