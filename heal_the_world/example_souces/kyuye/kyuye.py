import tensorflow as tf

data = [[1,2,3,4,5,6,7,8,9,10],
        [10,9,8,7,6,5,4,3,2,1],
        [1,2,3,4,5,6,7,8,9,10],
        [10,9,8,7,6,5,4,3,2,1]]

label = [1,0,1,0]

plh = tf.placeholder(dtype=tf.float32, shape=(4,10))
plh2 = tf.placeholder(dtype=tf.float32, shape=(4))

feed_dict = {plh : data, plh2 : label}

plh2 = tf.reshape(plh2,shape=(1,4))
var = tf.Variable(tf.random_normal(shape=(10,10)))
var2 = tf.Variable(tf.zeros(shape=(10)))

result = tf.matmul(plh,var) + var2
layer = tf.nn.sigmoid(result)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a = sess.run(layer,feed_dict)
    print(a)




