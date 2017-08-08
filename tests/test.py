
import tensorflow as tf
import numpy as np
import os
import urllib
import csv 

data = tf.contrib.data.TextLineDataset("./tests/test2.csv").skip(1).batch(10)
itr = data.make_one_shot_iterator()
batch = itr.get_next()

col0, col1, col2, col3, col4 = tf.decode_csv(batch, record_defaults=[[0]]*5)
batch = tf.stack([col0, col1, col2, col3, col4], axis=1)

with tf.Session() as sess:
    for i in range(10):
        print(sess.run(batch))
