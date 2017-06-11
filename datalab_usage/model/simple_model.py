from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import metrics
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

#
# 1. Input function
#
def input_fn(filename, batch_size=20):
  """
  We use this function to read the train data and build the batch
  If you have used to use the queue runners, go ahead at same
  Just care for the returns.
  you can see the return dictionary {'data': b_train}
  this form is for feed data from external query
  When you call the model to predict, you would use key for input name, value for input data to predict. That's why
  """
  filename_queue = tf.train.string_input_producer([filename])
  reader = tf.TextLineReader()
  _, value = reader.read(filename_queue)
  dataset = tf.decode_csv(value, [[0]]*11)
  label = tf.cast(dataset[0], tf.float32)
  train = tf.cast(tf.stack(dataset[1:]), tf.float32)

  b_train, b_label = tf.train.batch([train, label], batch_size)
  return {'data':b_train}, tf.reshape(b_label, (-1, 1))


def get_input_fn(filename, batch_size):
  """
  and you see the this function returns the input_fn as function type. 
  it's not a big deal. it's required by experiment function
  """
  return lambda: input_fn(filename, batch_size)


#
# 2. Estimator
#
def _simple_model_fn(features, labels, mode):
  """
  This function is just a model that you built
  you can see the input argument 'features'
  Actually it's a placeholders wrapped by dictionary 
  It's considered that any user who call this model inputs a data as a dict type to predict
  """
  x = features['data']
  layer1 = tf.layers.dense(x, 10)
  layer2 = tf.layers.dense(layer1, 10)
  layer3 = tf.layers.dense(layer2, 10)
  logits = tf.layers.dense(layer3, 1)

  loss = None
  train_op = None

  
  """
  Mode key means whether this model is training or not
  You can make a train_op and calculate loss like below
  """
  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    loss = tf.losses.sigmoid_cross_entropy(labels, logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=0.001, optimizer="Adam")

  """
  Then you can make the prediction form and return ModelFnOp object like below
  """
  # Generate Predictions
  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.sigmoid(logits)
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(mode=mode, loss=loss, train_op=train_op,
                               predictions=predictions)


def build_estimator(model_dir):
  """
  Lastly, build the estimator. Yeah just copy and paste. who knows?
  """
  return learn.Estimator(
         model_fn=_simple_model_fn,
         model_dir=model_dir,
         config=tf.contrib.learn.RunConfig(save_checkpoints_secs=180))


#
# 3. Evaluate metrics
#
def get_eval_metrics():
  """
  This function returns which metrics would like you see for evaluating
  These links would be help
  https://www.tensorflow.org/api_docs/python/tf/contrib/learn/MetricSpec
  https://www.tensorflow.org/api_docs/python/tf/metrics/accuracy
  """
  return {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy,
                                     prediction_key="classes")
         }

#
# 4. Serving function
#
def serving_input_fn():
  """
  You saw the placeholder dictionary like {'data': tensor } form. 
  'data' is the name of input. I will be query to model with input data like {'data': [0, 1, 2, 3..., 9]}
  You can make the number of inputs in dictionary here
  Make the feature_placeholders in dictionary like below, and copy and paste otherwise. who knows!!
  """
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