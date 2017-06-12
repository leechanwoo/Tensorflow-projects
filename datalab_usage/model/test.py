import tensorflow as tf
import os
import json
from pprint import pprint

# get environment variable TF_CONFIG to specify the cluster
tf_config = os.environ.get('TF_CONFIG')

tf_config_json = json.loads(tf_config)

cluster = tf_config_json.get('cluster')
job_name = tf_config_json.get('task', {}).get('type')
task_index = tf_config_json.get('task', {}).get('index')


# This is how to create server for distrubted learning
cluster_spec = tf.train.ClusterSpec(cluster)
server = tf.train.Server(cluster_spec,
                         job_name=job_name,
                         task_index=task_index)

data = [[i for i in range(10)], [9-i for i in range(10)]]
label = [[1], [0]]

# set ps listening
if job_name == 'ps':
  server.join()
elif job_name == 'worker':
  # seperate variables to the workers
  with tf.device(tf.train.replica_device_setter()):
    x = tf.placeholder(tf.float32, [None, 10])
    y_ = tf.placeholder(tf.float32, [None, 1])
    feed_dict = {x: data, y_: label}
    
    layer1 = tf.layers.dense(x, 10)
    layer2 = tf.layers.dense(layer1, 10)
    layer3 = tf.layers.dense(layer2, 10)
    logits = tf.layers.dense(layer3, 1)
    
    # you cannot use tf.Session() to train the model using for .. in ... 
    # so you need to create the object which define the global step
    global_step = tf.contrib.framework.get_or_create_global_step()
    
    loss = tf.losses.sigmoid_cross_entropy(y_, logits)
    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)

  # you can set the global step here
  hooks=[tf.train.StopAtStepHook(last_step=100)]
  
  # new session to distributed learning!
  # use this session following this code
  with tf.train.MonitoredTrainingSession(master=server.target,
                                         is_chief=(task_index == 0),
                                         checkpoint_dir="./model/train_logs",
                                         hooks=hooks) as mon_sess:
    while not mon_sess.should_stop():
      # Run a training step asynchronously.
      # See `tf.train.SyncReplicasOptimizer` for additional details on how to
      # perform *synchronous* training.
      # mon_sess.run handles AbortedError in case of preempted PS.
      mon_sess.run(train)