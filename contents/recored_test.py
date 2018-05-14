import tensorflow as tf

# this is neural network source

samples = 100
up = [i for i in range(10)]
down = [9-i for i in range(10)]

data =  [up if i%2 == 0 else down for i in range(samples)]
label = [[1] if i%2 == 0 else [0] for i in range(samples)]

x = tf.placeholder(tf.float32, [None, 10])
y_ = tf.placeholder(tf.float32)

layer1 = tf.layers.dense(x, 10)
layer2 = tf.layers.dense(layer1, 10)
layer3 = tf.layers.dense(layer2, 10)
layer4 = tf.layers.dense(layer3, 10)
out = tf.layers.dense(layer4, 1)

pred = tf.round(tf.nn.sigmoid(out))

accuracy = tf.metrics.accuracy(y_, pred)

loss = tf.losses.sigmoid_cross_entropy(y_, out)
train_op = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(100):
        _, _loss, _acc = sess.run([train_op, loss, accuracy],
                                  feed_dict={x: data, y_: label})
        print("")
        print('step: {}, loss: {}'.format(i, _loss))
        print('accuracy: {}'.format(_acc[0]))


    _pred = sess.run(pred, feed_dict={x: data[:3]})
    print('input data: {}'.format(data[:3]))
    print('prediction: {}'.format(_pred))
