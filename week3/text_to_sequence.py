# -*- coding: utf-8 -*-

"""
@Author: xiezizhe 
@Date: 2019/8/21 7:52 PM
"""
import tensorflow as tf

if __name__ == '__main__':
    sentences = [
        'I love my dog',
        'You love my dogs!',
        'Do you think my dog is amazing?'
    ]
    # the maximum number of words to keep, based on word frequency.
    # Only the most common `num_words-1` words will be kept.
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100, oov_token='<OOV>')
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(sentences)
    pad_seq = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="post", maxlen=6, truncating="post")

    print(word_index)
    print(sequences)
    print(pad_seq)

    test_data = ['my dog love my manatee']
    test_seq = tokenizer.texts_to_sequences(test_data)

    print(test_seq)
    print("Hello, world")
