"""
 * rnn tutorial
"""

import tensorflow as tf
import matplotlib.pyplot as plt

samples = 1000
hidden = 5
input_size = 1
state_size = 15

gen_x = tf.constant([0.01*i for i in range(samples + 1)], dtype=tf.float32)
gen_y = tf.sin(gen_x)

with tf.Session() as sess:
    gened_x, gened_y = sess.run([gen_x, gen_y])

plt.scatter(gened_x, gened_y, 1)
plt.savefig("./5_rnn/sine_tracker_target.png")

batch_shape = (int(samples/hidden), hidden, input_size )
batch_input = tf.reshape(gen_y[:-1], batch_shape)
batch_label = tf.reshape(gen_y[1:], batch_shape)

batch_set = [batch_input, batch_label]
x, y_ = tf.train.batch(batch_set, 50, enqueue_many=True)

print(x)
print(y_)

rnn_cell = tf.contrib.rnn.BasicRNNCell(state_size)
output, _ = tf.contrib.rnn.static_rnn(rnn_cell, tf.unstack(x, axis=1), dtype=tf.float32)

output_w = tf.Variable(tf.truncated_normal((hidden, state_size, input_size)))
output_b = tf.Variable(tf.zeros((input_size)))

y = tf.matmul(output, output_w) + output_b

loss = 0
for i in range(hidden):
    loss += tf.losses.mean_squared_error(tf.unstack(y_, axis=1)[i], y[i])
train = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)

test_y = tf.reshape(tf.transpose(tf.reduce_mean(y, axis=2)), (250,))


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess, coord)
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        _, _loss = sess.run([train, loss])
        if i % 100:
            print("loss: ", _loss)
    
    domain, prediction = sess.run([gen_x[:250], test_y])
    plt.scatter(domain, prediction, 2, 'r')
    plt.savefig("./5_rnn/sine_tracker_prediction.png")
    coord.request_stop()
    coord.join(thread)
