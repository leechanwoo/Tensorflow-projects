
import tensorflow as tf
import matplotlib.pyplot as plt

data = []
label = []
for i in range(1000):
    if i % 2 == 0:
        data.append(list(range(10)))
        label.append([1])
    else:
        data.append([10-i for i in range(10)])
        label.append([0])

x = tf.constant(data, tf.float32)
y_ = tf.constant(label, tf.float32)

Weight = tf.Variable(tf.truncated_normal(shape=(10,1)))
Bias = tf.Variable(tf.zeros(shape=(1)))

y = tf.nn.sigmoid(tf.matmul(x, Weight) + Bias)

loss = tf.losses.sigmoid_cross_entropy(y_, y)
train = tf.train.GradientDescentOptimizer(1e-10).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        _, _loss = sess.run([train, loss])
        print("step: %d, loss: %f"%(i, _loss))

    test = sess.run(y)

    print(data[:10])
    print(test[:10])
    print()