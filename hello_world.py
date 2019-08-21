# -*- coding: utf-8 -*-

"""
@Author: xiezizhe 
@Date: 2019/8/19 5:41 PM
"""
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    xs = np.reshape(np.array([-1, 0, 1, 2, 3, 4], dtype=float), [-1, 1])
    ys = np.reshape(np.array([-3, -1, 1, 3, 5, 7], dtype=float), [-1, 1])

    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.layers.dense(x, 1)
    loss = tf.losses.mean_squared_error(ys, y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    global_step = tf.Variable(0.0, trainable=False, name="global_step")
    train_op = optimizer.minimize(loss, global_step=global_step)

    epoch = 10000
    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

        for i in range(epoch):
            _, _loss, _ = sess.run([train_op, loss, global_step], feed_dict={
                x: xs
            })
            print("loss={}".format(_loss))

        res = sess.run([y], feed_dict={
            x: [[10]]
        })
        print(res)

    print("Hello, world")
