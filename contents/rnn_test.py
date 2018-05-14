import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sample = 1000
axis = np.array([0.01*float(i) for i in range(sample+1)], np.float32)
sin = np.sin(axis)

#plt.plot(y)
#plt.show()

reshaped_x = tf.reshape(sin[:-1], [-1, 5, 1])
reshaped_y = tf.reshape(sin[1:], [-1, 5, 1])

x = tf.unstack(reshaped_x, axis=1)
y_ = tf.unstack(reshaped_y, axis=1)

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=20)
output, static = tf.nn.static_rnn(cell, x, dtype=tf.float32)

output_w = tf.Variable(tf.truncated_normal(shape=[5, 20, 1]))
output_b = tf.Variable(tf.zeros(1))

y = tf.matmul(output, output_w) + output_b

loss = 0
for i in range(5):
    loss += tf.losses.mean_squared_error(y_[i], y[i])
train_op = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        _, _loss = sess.run([train_op, loss])
        print('loss: {}'.format(_loss))

    flat = tf.reshape(tf.transpose(y, [1, 0, 2]), shape=[1000,])
    pred = sess.run(flat)

    plt.plot(pred)
    plt.plot(sin[:-1])
    plt.show()
