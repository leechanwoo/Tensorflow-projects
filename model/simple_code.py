
# coding: utf-8

# # Datalab Usage part.2

# Hi guys, Welcome!

# Today, we're going to design the tensorflow model with datalab and run it using the GPU

# Unfortunately, datalab does not support GPU directly.

# So, we should use CloudML to run on GPU.

# now, Let's see how it makes

# 

# first, I prepare the simple code in tensorflow

# In[13]:

import tensorflow as tf

say_hello = tf.constant("hello datalab!!")

with tf.Session() as sess:
  print(sess.run(say_hello))
    
