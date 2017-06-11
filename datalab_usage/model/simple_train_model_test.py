from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import metrics
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

def input_fn(filename, batch_size=20):
  filename_queue = tf.train.string_input_producer([filename])
  reader = tf.TextLineReader()
  _, value = reader.read(filename_queue)
  dataset = tf.decode_csv(value, [[0]]*11)
  label = tf.cast(dataset[0], tf.float32)
  train = tf.cast(tf.stack(dataset[1:]), tf.float32)

  b_train, b_label = tf.train.batch([train, label], batch_size)
  return {'data':b_train}, tf.reshape(b_label, (-1, 1))
                    
def get_input_fn(filename, batch_size):
  return lambda: input_fn(filename, batch_size)

def _cnn_model_fn(features, labels, mode):
  x = features['data']
  layer1 = tf.layers.dense(x, 10)
  layer2 = tf.layers.dense(layer1, 10)
  layer3 = tf.layers.dense(layer2, 10)
  logits = tf.layers.dense(layer3, 1)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
      loss = tf.losses.sigmoid_cross_entropy(labels, logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
      train_op = tf.contrib.layers.optimize_loss(
          loss=loss,
          global_step=tf.contrib.framework.get_global_step(),
          learning_rate=0.001, optimizer="Adam")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.sigmoid(logits)
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(mode=mode, loss=loss, train_op=train_op,
                               predictions=predictions)


def build_estimator(model_dir):
  return learn.Estimator(
         model_fn=_cnn_model_fn,
         model_dir=model_dir,
         config=tf.contrib.learn.RunConfig(save_checkpoints_secs=180))


def get_eval_metrics():
  return {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy,
                                     prediction_key="classes")
         }


def serving_input_fn():
  feature_placeholders = {'data': tf.placeholder(tf.float32, [None, 10])}
  features = {
      key: tensor
      for key, tensor in feature_placeholders.items()
  }    
  return learn.utils.input_fn_utils.InputFnOps(
      features,
      None,
      feature_placeholders
  )