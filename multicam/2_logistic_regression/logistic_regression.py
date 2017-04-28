
import tensorflow as tf
import matplotlib.pyplot as plt

samples = 1000
data = [[float(i) * 0.01] for i in range(-samples,samples)]
label = [[0 if d[0] < 2.5 else 1] for d in data]

plt.scatter(data, label, 1)
plt.savefig("./logistic_regression/logistic_target.png")

x = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)

weight = tf.Variable(tf.zeros((1)))
bias = tf.Variable(tf.zeros((1)))
y = tf.nn.sigmoid( x * weight + bias )

loss = tf.losses.sigmoid_cross_entropy(y_, y)
loss = tf.Print(loss, [loss,weight, bias])
train = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        sess.run(train, {x:data, y_:label})
        if i % 100 == 0:
            print("step: ", i)
            valid_x, valid_y = sess.run([x, y], {x:data})
            if i > 1000:
                plt.scatter(valid_x, valid_y, 1, 'r')
                plt.savefig("./logistic_regression/logistic_prediction.png")
            elif i > 2000:
                plt.scatter(valid_x, valid_y, 1, 'b')
                plt.savefig("./logistic_regression/logistic_prediction.png")
            elif i > 3000:
                plt.scatter(valid_x, valid_y, 1, 'g')
                plt.savefig("./logistic_regression/logistic_prediction.png")
            elif i > 4000:
                plt.scatter(valid_x, valid_y, 1, 'r')
                plt.savefig("./logistic_regression/logistic_prediction.png")
            elif i > 5000:
                plt.scatter(valid_x, valid_y, 1, 'b')
                plt.savefig("./logistic_regression/logistic_prediction.png")
            elif i > 6000:
                plt.scatter(valid_x, valid_y, 1, 'g')
                plt.savefig("./logistic_regression/logistic_prediction.png")
            elif i > 7000:
                plt.scatter(valid_x, valid_y, 1, 'r')
                plt.savefig("./logistic_regression/logistic_prediction.png")
            elif i > 8000:
                plt.scatter(valid_x, valid_y, 1, 'b')
                plt.savefig("./logistic_regression/logistic_prediction.png")
            elif i > 9000:
                plt.scatter(valid_x, valid_y, 1, 'g')
                plt.savefig("./logistic_regression/logistic_prediction.png")
                