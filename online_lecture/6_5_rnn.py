
import tensorflow as tf

state_size = 100
time_step = 5

# [tensor, tensor, tensor, ... , tensor]
#input_tensor = [tf.constant(), tf.constant(2), tf.constant(3), tf.constant(4), tf.constant(5)]
input_tensor = [
    tf.constant(
        [ [i] for j in range(100)], dtype=tf.float32) 
        for i in range(time_step)
]

for i in input_tensor:
    print(i)

cell = tf.contrib.rnn.BasicRNNCell(state_size)
output, state = tf.contrib.rnn.static_rnn(cell, input_tensor, dtype=tf.float32)