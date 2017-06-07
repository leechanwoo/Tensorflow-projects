
import tensorflow as tf

say_hello = tf.constant("hello datalab!!")

with tf.Session() as sess:
  print(sess.run(say_hello))