
import tensorflow as tf
import time

tic = time.clock()

mat_size = (1000, 1000)
vec_size = (1000, 1)
bias_size = (1000,)

x = tf.random_normal(mat_size)
y_ = tf.random_normal(vec_size)

w1 = tf.Variable(tf.random_normal(mat_size))
b1 = tf.Variable(tf.random_normal(bias_size))
layer1 = tf.nn.sigmoid( tf.matmul(x, w1) + b1 )

w2 = tf.Variable(tf.random_normal(mat_size))
b2 = tf.Variable(tf.random_normal(bias_size))
layer2 = tf.nn.sigmoid( tf.matmul(layer1, w2) + b2 )

w3 = tf.Variable(tf.random_normal(mat_size))
b3 = tf.Variable(tf.random_normal(bias_size))
layer3 = tf.nn.sigmoid( tf.matmul(layer2, w3) + b3 )

w4 = tf.Variable(tf.random_normal(mat_size))
b4 = tf.Variable(tf.random_normal(bias_size))
layer4 = tf.nn.sigmoid( tf.matmul(layer3, w4) + b4 )

w5 = tf.Variable(tf.random_normal(mat_size))
b5 = tf.Variable(tf.random_normal(bias_size))
layer5 = tf.nn.sigmoid( tf.matmul(layer4, w5) + b5 )

w6 = tf.Variable(tf.random_normal(mat_size))
b6 = tf.Variable(tf.random_normal(bias_size))
layer6 = tf.nn.sigmoid( tf.matmul(layer5, w6) + b6 )

w7 = tf.Variable(tf.random_normal(mat_size))
b7 = tf.Variable(tf.random_normal(bias_size))
layer7 = tf.nn.sigmoid( tf.matmul(layer6, w7) + b7 )

w8 = tf.Variable(tf.random_normal(mat_size))
b8 = tf.Variable(tf.random_normal(bias_size))
layer8 = tf.nn.sigmoid( tf.matmul(layer7, w8) + b8 )

w9 = tf.Variable(tf.random_normal(mat_size))
b9 = tf.Variable(tf.random_normal(bias_size))
layer9 = tf.nn.sigmoid( tf.matmul(layer8, w9) + b9 )

w10 = tf.Variable(tf.random_normal(mat_size))
b10 = tf.Variable(tf.random_normal(bias_size))
layer10 = tf.nn.sigmoid( tf.matmul(layer9, w10) + b10 )

w11 = tf.Variable(tf.random_normal(mat_size))
b11 = tf.Variable(tf.random_normal(bias_size))
layer11 = tf.nn.sigmoid( tf.matmul(layer10, w11) + b11 )


wo = tf.Variable(tf.random_normal(vec_size))
bo = tf.Variable(0.0)
y = tf.matmul(layer11, wo,) + bo 

loss = tf.losses.mean_squared_error(y_, y)
train = tf.train.AdamOptimizer(1.0).minimize(loss)

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
    for i in range(1000):
        print("step: ", i)
        sess.run(train)

toc = time.clock()

print("process time: ", toc - tic, "s")