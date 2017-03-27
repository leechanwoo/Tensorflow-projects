import tensorflow as tf

data = [[1,2,3,4,5,6,7,8,9,10],
        [10,9,8,7,6,5,4,3,2,1],
        [1,2,3,4,5,6,7,8,9,10],
        [10,9,8,7,6,5,4,3,2,1]]

label = [1,0,1,0]

plh_data = tf.placeholder(dtype=tf.float32, shape=(4,10))
plh_label = tf.placeholder(dtype=tf.float32, shape=(4))

feed_dict = {plh_data : data, plh_label : label}

#plh2 = tf.reshape(plh2,shape=(1,4))
weight = tf.Variable(tf.random_normal(shape=(10,1)))
bias = tf.Variable(tf.zeros(shape=(1)))

result = tf.matmul(plh_data,weight) + bias
layer = tf.nn.sigmoid(result)

cost_function = -plh_label*tf.log(layer) - (1-plh_label)*tf.log(1-layer)
cost_mean = tf.reduce_mean(cost_function)
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost_mean)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a = sess.run(layer,feed_dict)
    output = sess.run(train)
    cost_output = sess.run(cost_function)
    print(a,output,cost_output)






