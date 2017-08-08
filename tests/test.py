
import tensorflow as tf
import numpy as np
import os
import urllib
import csv 

data = tf.contrib.data.TextLineDataset("./tests/test2.csv").skip(1).batch(10)
itr = data.make_one_shot_iterator()
batch = itr.get_next()

cols = tf.decode_csv(batch, record_defaults=[[0]]*5)

batch = tf.stack(cols, axis=1)

with tf.Session() as sess:
    for i in range(10):
        print(sess.run(batch))
