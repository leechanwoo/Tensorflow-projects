#for i, j in enumerate()

import tensorflow as tf 
from tensorflow.python import debug as tf_debug

a = tf.constant([1, 2, 3])
b = tf.constant([5, 6, 5, 4, 4])

a = tf.Print(a, [a, b], message="a, b ", first_n=3, summarize=5)

c = a + b
a = a + c

with tf.Session() as sess:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    for _ in range(10):
        result = sess.run(a)
    print(result)