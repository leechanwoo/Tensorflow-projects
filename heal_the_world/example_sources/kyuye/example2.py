import tensorflow as tf

data = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]

label = [1, 0, 1, 0]

x = tf.placeholder(tf.float32, shape=(4, 10))
y_ = tf.placeholder(tf.float32, shape=(4))
feed_dict = {x: data, y_: label}

y_ = tf.reshape(y_, (4, 1))

# set variables
layer1_w = tf.Variable(tf.random_normal((10, 10)), dtype=tf.float32)
layer1_b = tf.Variable(tf.zeros(10), dtype=tf.float32)

layer2_w = tf.Variable(tf.random_normal((10, 10)), dtype=tf.float32)
layer2_b = tf.Variable(tf.zeros(10), dtype=tf.float32)

out_w = tf.Variable(tf.random_normal((10, 1)), dtype=tf.float32)
out_b = tf.Variable(tf.zeros(1), dtype=tf.float32)


# model
layer1 = tf.nn.sigmoid(tf.matmul(x, layer1_w) + layer1_b)
layer2 = tf.nn.sigmoid(tf.matmul(layer1, layer2_w) + layer2_b)
y = tf.nn.sigmoid(tf.matmul(layer2, out_w) + out_b)


# cost = -y_*tf.log(y) - (1-y_)*tf.log(1-y)
# cost_mean = tf.reduce_mean(cost)
# train = tf.train.GradientDescentOptimizer(0.01).minimize(cost_mean)

# init = tf.global_variables_initializer()

# default saver form
saver = tf.train.Saver()  #생성된 그래프 중에 variables 그래프만 모음

#saver = tf.train.Saver( [var1, var2, var3] )
#var1, var2, var3만 check point 저장


with tf.Session() as sess:
#    sess.run(init)
    saver.restore(sess, './test')
    # print()
    # print("training start")

    # for i in range(1000):
    #     _, _cost, _cost_mean, _pred, _lab = sess.run([train, cost, cost_mean, y, y_], feed_dict)
    #     if i % 50 == 0:
    #         print("training error: ", _cost_mean)
    # path = saver.save(sess, './test')

    for i in range(10):
        _pred = sess.run(y, feed_dict)
        print(_pred)
        print()

    # print()
    # print("cost of each data: ")
    # print(_cost)
    # print()
    # print("mean of cost: ", _cost_mean)
    # print()
    # print("training end")
    # print()
    # print("prediction")
    # print(_pred)
    # print()
    # print("label")
    # print(_lab)
    # print()
    for i in range(4):
        print("data[%d] "%(i), data[i], " is ", "increasing" if _pred[i] > 0.5 else "decreasing")
