# -*- coding: utf-8 -*-

"""
@Author: xiezizhe 
@Date: 2019/10/14 7:59 PM
"""

import tensorflow as tf
from tensorflow.python.keras.utils import to_categorical
import tensorflow_text as text

print(tf.__version__)


class FastText(tf.keras.Model):

    def __init__(self, max_feature, embedding_dim, max_len, n_class=1, activation='sigmoid'):
        super(FastText, self).__init__()
        self.max_feature = max_feature
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.n_class = n_class
        self.activation = activation
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.max_feature, self.embedding_dim, input_length=self.max_len),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(self.n_class, activation=self.activation)
        ])
        self.model.summary()

    def call(self, inputs):
        """
        :param inputs:  [batch_size, number of steps, number of Vocabulary]
        :return:
        """
        return self.model(inputs)


if __name__ == '__main__':
    embedding_dim = 64
    max_len = 64
    max_feature = 2048
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_feature)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len, padding="post", truncating="post")
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len, padding="post", truncating="post")
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    model = FastText(max_feature, embedding_dim, max_len, 1, 'sigmoid')
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(0.001)

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
    for epoch in range(1):
        for (batch, (x, y)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                predictions = model(x)
                loss = tf.reduce_mean(loss_object(y, predictions))
            print(loss.numpy())
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
