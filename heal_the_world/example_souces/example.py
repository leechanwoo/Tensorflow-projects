import tensorflow as tf

const = tf.constant("hello world")
var = tf.Variable(22)
plh = tf.placeholder(dtype=tf.int32)

with tf.Session() as sess:
    print(sess.run(const))

    sess.run(tf.global_variables_initializer())
    print(sess.run(var))
    
    print(sess.run(plh,{plh:30}))
    
    




