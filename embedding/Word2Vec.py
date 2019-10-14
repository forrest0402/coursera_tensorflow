# -*- coding: utf-8 -*-

"""
@Author: xiezizhe 
@Date: 2019/10/14 7:59 PM
"""

import tensorflow as tf
import tensorflow_text as text
import numpy as np
import time
import matplotlib.pyplot as plt

print(tf.__version__)
MAX_LEN = 2
BATCH_SIZE = 20
NUM_SAMPLED = 10

tf.enable_eager_execution()


class Word2Vec(tf.keras.Model):

    def __init__(self, voc_size, embedding_size):
        super(Word2Vec, self).__init__()
        self.voc_size = voc_size
        self.embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
        self.nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
        self.nce_biases = tf.Variable(tf.zeros([voc_size]))

    def cost(self, labels, inputs, num_sampled):
        return tf.reduce_mean(
            tf.nn.nce_loss(self.nce_weights, self.nce_biases, labels, inputs, num_sampled, self.voc_size))

    def call(self, inputs):
        """
        :param inputs:  [batch_size, number of steps, number of Vocabulary]
        :return:
        """
        selected_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        return selected_embed


def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = word2vec(x)
        loss = word2vec.cost(y, predictions, NUM_SAMPLED)

    gradients = tape.gradient(loss, word2vec.trainable_variables)
    optimizer.apply_gradients(zip(gradients, word2vec.trainable_variables))


def make_data(sentences, word_index):
    skip_grams = []
    tokenizer = text.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(sentences)
    ngrams = text.ngrams(tokens, 3, reduction_type=text.Reduction.STRING_JOIN)
    segments = np.array([x[0].decode("UTF-8").split(" ") for x in ngrams.to_list()])
    for segment in segments:
        skip_grams.append([segment[1], segment[0]])
        skip_grams.append([segment[1], segment[-1]])

    return np.vectorize(lambda x: word_index[x] - 1)(skip_grams)


def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])  # target
        random_labels.append([data[i][1]])  # context word

    return np.array(random_inputs, dtype=np.int64), tf.cast(random_labels, dtype=tf.int64)


if __name__ == '__main__':
    sentences = ["i like dog", "i love coffee", "i hate milk",
                 "i like animal",
                 "dog cat animal", "apple cat dog like", "dog fish milk like",
                 "dog cat eyes like", "i like apple", "apple i hate",
                 "apple i movie book music like", "cat dog hate", "cat dog like"]
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100, oov_token='<OOV>')
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    voc_size = len(word_index)

    skip_grams = make_data(sentences, word_index)
    word2vec = Word2Vec(voc_size, MAX_LEN)
    optimizer = tf.keras.optimizers.Adam(0.001)

    for epoch in range(5000):
        start = time.time()
        batch_inputs, batch_labels = random_batch(skip_grams, BATCH_SIZE)
        train_step(batch_inputs, batch_labels)
        print('Time taken for {} epoch: {} secs\n'.format(epoch, time.time() - start))

    for (label, i) in word_index.items():
        x, y = word2vec.embeddings[i - 1]
        plt.scatter(x.numpy(), y.numpy())
        plt.annotate(label, xy=(x.numpy(), y.numpy()), xytext=(5, 2), textcoords='offset points', ha='right',
                     va='bottom')
    plt.show()
