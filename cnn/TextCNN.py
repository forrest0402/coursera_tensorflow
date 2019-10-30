# -*- coding: utf-8 -*-

"""
@Author: xiezizhe 
@Date: 2019/10/15 4:25 PM
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

print(tf.__version__)


class TextCNNLayer(tf.keras.layers.Layer):

    def __init__(self, filters, filter_depth, emb_dim):
        super(TextCNNLayer, self).__init__()
        self.filter_list = []
        for i, filter_size in enumerate(filters):
            region = self.region(filter_size, filter_depth, emb_dim)
            self.filter_list.append(region)

    def call(self, input_tensor):
        branches = []
        for region_filter in self.filter_list:
            branches.append(region_filter(input_tensor))

        return tf.keras.layers.Flatten()(tf.keras.layers.Concatenate(axis=1)(branches))

    def region(self, filter_size, filter_depth, emb_dim):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filter_depth, [filter_size, emb_dim], activation=tf.nn.tanh, padding="same"),
            tf.keras.layers.GlobalMaxPooling2D()
        ])


class TextCNN(tf.keras.Model):

    def __init__(self, vocab_size, max_len, emb_dim, filter_depth, output_size,
                 filters=[2, 2, 3, 3, 5, 5, 7, 7],
                 rate=0.5,
                 task_output=tf.keras.layers.Dense(1, activation='sigmoid')):
        super(TextCNN, self).__init__()

        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, emb_dim, input_length=max_len),
            tf.keras.layers.Reshape((max_len, emb_dim, 1)),
            TextCNNLayer(filters, filter_depth, emb_dim),
            tf.keras.layers.Dense(output_size),
            tf.keras.layers.Dropout(rate),
            task_output
        ])
        self.model.summary()

    def call(self, inputs, training):
        return self.model(inputs, training=training)


if __name__ == '__main__':
    embedding_dim = 64
    max_len = 64
    max_feature = 2048
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_feature)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len, padding="post", truncating="post")
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len, padding="post", truncating="post")
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    model = TextCNN(vocab_size=max_feature,
                    max_len=max_len,
                    emb_dim=embedding_dim,
                    filter_depth=100,
                    output_size=8196,
                    task_output=tf.keras.layers.Dense(1, activation='sigmoid'))

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(0.001)
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)
    for epoch in range(1):
        losses = []
        for (batch, (x, y)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                predictions = model(x, training=True)
                loss = tf.reduce_mean(loss_object(y, predictions))
            losses.append(loss.numpy())
            if batch % 100 == 0:
                print("{}/{} - loss: {}".format(batch, len(x_train) / 32, np.mean(losses)))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        accu = []
        for (batch, (x, y)) in enumerate(test_dataset):
            predictions = tf.round(model(x, training=False))
            accu.append(
                tf.reduce_mean(tf.cast(tf.equal(predictions, tf.cast(y, dtype=tf.float32)), dtype=tf.float32)).numpy())
            print(np.mean(accu))
