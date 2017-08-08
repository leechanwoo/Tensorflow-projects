
import tensorflow as tf
import matplotlib.pyplot as plt

samples = 1000
data = [[float(i) * 0.01] for i in range(-samples,samples)]
label = [[0 if d[0] < 2.5 else 1] for d in data]

savepng = "multicam/2_logistic_regression/logistic_prediction.png"
path = "multicam/2_logistic_regression/"

plt.scatter(data, label, 1)
plt.savefig(savepng)

x = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)

weight = tf.Variable(-0.5)
bias = tf.Variable(tf.zeros((1)))
y =  x * weight + bias

# saver = tf.train.Saver()


loss = tf.losses.sigmoid_cross_entropy(y_, y)
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph(path+"model.ckpt.meta")
    saver.restore(sess, path+"model.ckpt")
    for i in range(10000):
        _, _loss, valid_x, valid_y = sess.run([train, loss, x, tf.sigmoid(y)], {x:data, y_:label})

        if i % 10 == 0:
            if i < 1000:
                plt.scatter(valid_x, valid_y, 1, 'r')
                plt.savefig(savepng)


   