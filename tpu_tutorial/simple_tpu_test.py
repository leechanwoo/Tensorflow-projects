import tensorflow as tf

test = tf.constant('tpu test')

with tf.Session() as sess:
    _test = sess.run(test)
    print(_test)
