
import tensorflow as tf
import matplotlib.pyplot as plt


data = []
for i in range(1000):
    data.append(list(range(i, 10+i)))

label = list(range(10, 1009))

x = tf.constant(data, tf.float32)
y_ = tf.constant(label, tf.float32)

exit()


x = tf.constant(list(range(1000)), tf.float32)
y = 0.1 * x + 5.8 + tf.random_normal(shape=(1000,), stddev=100)
target = 0.1 * x + 5.8

with tf.Session() as sess:
    _x, _y, _t = sess.run([x, y, target])
    plt.scatter(_x, _y, 1)
    plt.scatter(_x, _t, 2)

# ax + b ==> 1x + 0
a = tf.Variable(1, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

pred = a * x + b

loss = tf.losses.mean_squared_error(y, pred)
train = tf.train.GradientDescentOptimizer(1e-7).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(100):
        _, _loss, _pred = sess.run([train, loss, pred])

        plt.scatter(_x, _pred, 2, 'r')

        print("step: ", i, " loss: ", _loss)

    
    plt.savefig('./online_lecture/linear_reg_example.png')