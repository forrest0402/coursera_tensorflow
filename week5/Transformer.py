# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

"""
@Author: xiezizhe 
@Date: 2019/8/30 11:11 AM
"""
BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_LENGTH = 40

if __name__ == '__main__':
    print(tf.__version__)
    tf.enable_eager_execution()
    # print(tfds.list_builders())

    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus((en.numpy() for pt, en in train_examples),
                                                                           target_vocab_size=2 ** 13)

    tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus((pt.numpy() for pt, en in train_examples),
                                                                           target_vocab_size=2 ** 13)

    sample_string = 'Transformer is awesome.'
    print(sample_string)
    tokenized_string = tokenizer_en.encode(sample_string)
    print('Tokenized string is {}'.format(tokenized_string))

    original_string = tokenizer_en.decode(tokenized_string)
    print('The original string: {}'.format(original_string))

    assert original_string == sample_string
