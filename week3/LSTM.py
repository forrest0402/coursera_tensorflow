# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

"""
@Author: xiezizhe
@Date: 2019/8/29 4:50 PM
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

print(tf.__version__)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


if __name__ == '__main__':
    tf.enable_eager_execution()
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    tokenizer = info.features['text'].encoder

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, train_dataset.output_shapes)
    test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, activation="relu")),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation="relu")),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.summary()

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    NUM_EPOCHS = 10
    history = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)

    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
    print("hello, world")
