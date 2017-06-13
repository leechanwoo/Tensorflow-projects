
import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils

def serving_input_fn():
    feature_placeholders = {
        "hello": tf.placeholder(tf.string, [None])
    }

    features = {
        key: tf.expand_dims(tensor, -1) 
        for key, tensor in feature_placeholders.items()
    }

    return tflearn.utils.input_fn_utils.InputFnOps(
        features,
        None,
        feature_placeholders)


x = tf.constant([list(range(10)) for _ in range(100)])
layer1 = tf.layers.dense(x, 10)
layer2 = tf.layers.dense(layer1, 10)
layer3 = tf.layers.dense(layer2, 10)
out = tf.layers.dense(layer3, 1, activation=tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(tf.ones_like(out), out)
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

tf.saved_model.main_op.main_op