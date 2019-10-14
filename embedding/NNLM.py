# -*- coding: utf-8 -*-

"""
@From: 'Bengio, Yoshua, et al. "A neural probabilistic language model." Journal of machine learning research 3.Feb (2003): 1137-1155.'
@Author: xiezizhe 
@Date: 2019/10/12 10:42 AM
@Refer https://www.tensorflow.org/tutorials/text/transformer https://github.com/graykode/nlp-tutorial/blob/master/1-1.NNLM/NNLM_Tensor.ipynb
"""
import tensorflow as tf
import tensorflow_text as text
import numpy as np
import time
from tensorflow.python.keras.utils import to_categorical

print(tf.__version__)
MAX_LEN = 2
BATCH_SIZE = 3

tf.enable_eager_execution()


class NNLM(tf.keras.Model):

    def __init__(self, n_step, n_hidden, n_class):
        super(NNLM, self).__init__()
        self.n_step = n_step
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.n_class, self.n_hidden, input_length=MAX_LEN),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.n_hidden, activation=tf.nn.tanh),
            tf.keras.layers.Dense(self.n_class, activation=tf.nn.softmax)
        ])
        self.model.summary()

    def call(self, inputs):
        """
        :param inputs:  [batch_size, number of steps, number of Vocabulary]
        :return:
        """
        return self.model(inputs)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')


def loss_function(real, pred):
    loss = loss_object(real, pred)
    return tf.reduce_mean(loss)


def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = sample_nnlm(x)
        loss = loss_function(y, predictions)

    gradients = tape.gradient(loss, sample_nnlm.trainable_variables)
    optimizer.apply_gradients(zip(gradients, sample_nnlm.trainable_variables))

    train_loss(loss)
    train_accuracy(y, predictions)


def make_data(sentences, window_size):
    tokenizer = text.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(sentences)
    ngrams = text.ngrams(tokens, window_size + 1, reduction_type=text.Reduction.STRING_JOIN)
    segments = np.array([x[0].decode("UTF-8").split(" ") for x in ngrams.to_list()])
    input_batch = [' '.join(x) for x in segments[:, 0:-1]]
    target_batch = to_categorical(np.vectorize(lambda x: word_index[x] - 1)(segments[:, -1]), n_class, dtype='float32')
    return input_batch, target_batch


if __name__ == '__main__':
    sentences = ["i like dog", "i love coffee", "i hate milk", "I am zizhe"]
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100, oov_token='<OOV>')
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    n_class = len(word_index)
    n_step, n_hidden = MAX_LEN, 10

    x, y = make_data(sentences, n_step)
    x = tokenizer.texts_to_sequences(x)
    x = tf.keras.preprocessing.sequence.pad_sequences(x, padding="post", maxlen=MAX_LEN, truncating="post")
    sample_nnlm = NNLM(n_step, n_hidden, n_class)
    learning_rate = CustomSchedule(n_hidden)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(model=sample_nnlm, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    for epoch in range(100):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        train_step(x, y)
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
