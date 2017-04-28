import tensorflow as tf

"""
tensor = tf.constant([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])

print(tensor)
print(type(tensor))
"""


tens1 = tf.constant([[1, 2, 3]])
tens2 = tf.constant([[1], [2], [3]])
tens3 = tf.constant(2)

tens4 = tf.matmul(tens1, tens2) + tens3

merge = tf.summary.merge_all()
writer = tf.summary.FileWriter("./tensorboard/")

with tf.Session() as sess:
    _merge, _ = sess.run([merge, tens4])

