# -*- coding: utf-8 -*-

"""
@From: 'Bengio, Yoshua, et al. "A neural probabilistic language model." Journal of machine learning research 3.Feb (2003): 1137-1155.'
@Author: xiezizhe 
@Date: 2019/10/12 10:42 AM
"""
import tensorflow as tf
import numpy as np


class NNLM(tf.keras.layers.Layer):
    def __init__(self, n_step, n_hidden):
        super(NNLM, self).__init__()
        self.n_step = n_step
        self.n_hidden = n_hidden

    def call(self, inputs):
        pass


if __name__ == '__main__':
    sentences = ["i like dog", "i love coffee", "i hate milk"]
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100, oov_token='<OOV>')
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(sentences)
    pad_seq = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="post", maxlen=6, truncating="post")
