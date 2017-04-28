
import tensorflow as tf
import matplotlib.pyplot as plt

samples = 1000
data = tf.constant( 
    [[0.1*i for i in range(10)] if row % 2 == 0 else [0.1*(10-i) for i in range(10)] 
    for row in range(samples)], tf.float32) + tf.random_uniform((samples, 10), -0.05, 0.05)
    
label = tf.constant([ [1] if i % 2 == 0 else [0] for i in range(samples)])

with tf.Session() as sess:
    data, label = sess.run([data, label])

x = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)
step = tf.placeholder(tf.int32)

weight_1 = tf.Variable(tf.truncated_normal((10, 20)))
bias_1 = tf.Variable(tf.zeros(20))
layer_1 = tf.nn.sigmoid( tf.matmul(x, weight_1) + bias_1 )

weight_2 = tf.Variable(tf.truncated_normal((20, 20)))
bias_2 = tf.Variable(tf.zeros(20))
layer_2 = tf.nn.sigmoid( tf.matmul(layer_1, weight_2) + bias_2 )

weight_3 = tf.Variable(tf.truncated_normal((20, 1)))
bias_3 = tf.Variable(tf.zeros(1))
y = tf.nn.sigmoid( tf.matmul(layer_2, weight_3) + bias_3 )

loss = tf.losses.sigmoid_cross_entropy(y_, y)
loss = tf.Print(loss, [step, loss, y_[:2], y[:2]], message='step, loss, y_, y')
train = tf.train.GradientDescentOptimizer(1e-1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {x: data, y_: label, step: i})

    print(["increasing" if i > 0.5 else "decreasing" for i in sess.run(y, {x: data})[:20]])
    