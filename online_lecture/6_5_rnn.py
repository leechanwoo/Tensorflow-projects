
import tensorflow as tf

state_size = 20
input_tensor = tf.constant([
    [ 0, 0, 0, 0, 0],
    [ 1, 1, 1, 1, 1],
    [ 2, 2, 2, 2, 2],
    [ 3, 3, 3, 3, 3],
    [ 4, 4, 4, 4, 4]])

inputs = tf.unstack(input_tensor)

print(inputs)

cell = tf.contrib.rnn.BasicRNNCell(state_size)
output, state = tf.contrib.rnn.static_rnn(cell, inputs, dtype=tf.float32)



