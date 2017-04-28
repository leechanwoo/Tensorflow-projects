
import tensorflow as tf
import matplotlib.pyplot as plt

samples = 1000
data = tf.constant([i for i in range(samples)], tf.float32)
label = 0.2 * data + 2.4 + tf.random_normal((samples,), stddev=70)
target_model = 0.2 * data + 2.4

data = tf.reshape(data, (samples, 1))
label = tf.reshape(label, (samples,1))

with tf.Session() as sess:
    data, label, target_model = sess.run([data, label, target_model])
    
plt.scatter(data, label, 1)
plt.scatter(data, target_model, 1)
plt.savefig("./1_linear_regression/linear_regression_target.png")

x = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)

weight = tf.Variable(tf.truncated_normal((1,)))
bias = tf.Variable(tf.zeros((1)))
y = x * weight + bias

loss = tf.losses.mean_squared_error(y_, y)

loss = tf.Print(loss, [loss])

train = tf.train.GradientDescentOptimizer(1e-6).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        if i < 10:
            valid_x, valid_y = sess.run([x, y], {x:data})
            plt.scatter(valid_x, valid_y, 2, 'r')

        sess.run(train, {x: data, y_: label})


plt.savefig("./1_linear_regression/linear_regression_target.png")
