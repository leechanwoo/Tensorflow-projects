
import tensorflow as tf
import matplotlib.pyplot as plt

data = []
label = []
for i in range(1000):
    if i % 2 == 0:
        data.append([float(i-5) * 0.1 for i in range(10)])
        label.append([1])
    else:
        data.append([float(5-i) * 0.1 for i in range(10)])
        label.append([0])

        
x = tf.constant(data, tf.float32)
y_ = tf.constant(label, tf.float32)

w1 = tf.Variable(tf.truncated_normal(shape=(10,20)))
b1 = tf.Variable(tf.zeros(shape=(20)))
layer1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.truncated_normal(shape=(20,10)))
b2 = tf.Variable(tf.zeros(shape=(10)))
layer2 = tf.nn.sigmoid(tf.matmul(layer1, w2) + b2)

wo = tf.Variable(tf.truncated_normal(shape=(10,1)))
bo = tf.Variable(tf.zeros(shape=(1)))
pred = tf.nn.sigmoid(tf.matmul(layer2, wo) + bo)

loss = tf.losses.sigmoid_cross_entropy(y_, pred)
train = tf.train.GradientDescentOptimizer(1e-1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        _, _loss = sess.run([train, loss])
        if i % 100 == 0:
            print("step: %d, loss: %f"%(i, _loss))

    test = sess.run(pred)

    print(data[:10])
    print(test[:10])
    print()