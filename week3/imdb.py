# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 2019/8/28 7:52 PM
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

print(tf.__version__)
if __name__ == '__main__':
    tf.enable_eager_execution()
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    train_data, test_data = imdb['train'], imdb['test']

    training_sentences = []
    training_labels = []

    testing_sentences = []
    testing_labels = []

    for (x, y) in train_data:
        training_sentences.append(str(x.numpy()))
        training_labels.append(str(y.numpy()))

    for (x, y) in test_data:
        testing_sentences.append(str(x.numpy()))
        testing_labels.append(str(y.numpy()))

    training_labels = np.array(training_labels)
    testing_labels = np.array(testing_labels)

    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    oov_tok = "<OOV>"

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(training_sentences)
    train_padded_seq = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_length,
                                                                     truncating=trunc_type)

    test_sequences = tokenizer.texts_to_sequences(testing_sentences)
    test_padded_seq = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_length,
                                                                    truncating=trunc_type)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    num_epochs = 10
    model.fit(train_padded_seq, training_labels, epochs=num_epochs, validation_data=(test_padded_seq, testing_labels))
