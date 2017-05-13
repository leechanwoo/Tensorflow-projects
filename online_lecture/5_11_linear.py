
import tensorflow as tf
import matplotlib.pyplot as plt


data = []
for i in range(1000):
    data.append(list(range(i, 10+i)))

label = list(range(10, 1010))

x = tf.constant(data, tf.float32)
y_ = tf.constant(label, tf.float32)

y_ = tf.reshape(y_, (1000, 1))

Weight = tf.Variable(tf.truncated_normal(shape=(10,1)))
Bias = tf.Variable(tf.zeros(shape=(1)))

y = tf.matmul(x, Weight) + Bias 

loss = tf.losses.mean_squared_error(y_, y)
train = tf.train.GradientDescentOptimizer(1e-8).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        _, _loss = sess.run([train, loss])
        if i % 100 == 0:
            print("step: %d, loss: %f"%(i, _loss))

    test = sess.run(y)

    for i in range(10):
        print(data[i+50])
        print(test[i+50])
        print()