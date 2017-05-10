# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random

x_data = np.array([[random.uniform (-1, 1),random.uniform (-1, 1)], [random.uniform (-1, 1),random.uniform (-1, 1)], [random.uniform (-1, 1),random.uniform (-1, 1)], [random.uniform (-1, 1),random.uniform (-1, 1)]])
y_data = np.array([[0],[1],[1],[0]])

def generatePoints(numberOfPoints):
    
    trainingset = []

    for i in range (numberOfPoints):
        random.seed(1)
        x = random.uniform (-1, 1)
        y = random.uniform (-1, 1)
        trainingset.append([x,y])
    return trainingset

def classficationY(trainingset):
    y = []
    for i in range(50):
        target_function = trainingset[i][0] - trainingset[i][1]
        target_function_2 = 1 - (trainingset[i][0] * 2) - (trainingset[i][1] * 4)
        
        if target_function > 0 and target_function_2 > 0:
            y.append([-1])
        elif target_function < 0 and target_function_2 < 0:
            y.append([-1])
        elif target_function > 0 and target_function_2 < 0:
            y.append([1])
        elif target_function < 0 and target_function_2 > 0:
            y.append([1])
    return y
    
training_set = generatePoints(50)
y = classficationY(training_set)


x_ = tf.placeholder(tf.float32, shape=[50,2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[50,1], name="y-input")

Theta1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="Theta1")
Theta2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name="Theta2")

Bias1 = tf.Variable(tf.ones([2]), name="Bias1")
Bias2 = tf.Variable(tf.ones([1]), name="Bias2")

A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)
Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

a = ( (y_ * tf.log(Hypothesis)) + ((1 - y_) * tf.log(1.0 - Hypothesis)) ) * -1

#cost = tf.reduce_mean(( (y_ * tf.log(Hypothesis)) + ((1 - y_) * tf.log(1.0 - Hypothesis)) ) * -1)
cost = -y_*tf.log(Hypothesis) - (1-y_)*tf.log(1-Hypothesis)
cost_mean = tf.reduce_mean(cost)
#cost = tf.losses.sigmoid_cross_entropy(y_, Hypothesis)

train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cost_mean)


XOR_X = training_set
XOR_Y = y

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(100000):
    sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
    if i % 1000 == 0:
        print(sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
        # print('Epoch ', i)
        # print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))
        # print('Theta1 ', sess.run(Theta1))
        # print('Bias1 ', sess.run(Bias1))
        # print('Theta2 ', sess.run(Theta2))
        # print('Bias2 ', sess.run(Bias2))
        # print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
        # print('shaoe:',a)