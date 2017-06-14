import tensorflow as tf
from tensorflow.contrib.learn import Experiment
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn import Estimator
from tensorflow.contrib.learn import ModeKeys

import os
import json
import argparse

tf.logging.set_verbosity(tf.logging.INFO)

# all the function would be retured as function type
# so I also define the train_input function inside of train_input_fn function 
# that returns the trian_input as function type
def train_input_fn(args):
    def train_input():
        # genertate the train data (10000, 10) and label data (10000, 1)
        data = []
        label = []
        for i in range(100):
            if i % 2 == 0:
                # label 1 at increasing data 
                data.append([i for i in range(10)])
                label.append([1])
            else:
                # label 0 at decreasing data
                data.append([9-i for i in range(10)])
                label.append([0])

        return tf.constant(data, tf.float32), tf.constant(label, tf.float32)
    return train_input

# your model whatever you want
# you can see the input arguments of function pair
# I set the input argument of args which is got from shell input to model_fn input
# so I feel free to use it in the function
# and 3 arguments are in the model
# It's to be called with 3 arguments by experiment object 
def model_fn(args):
    def model(data, label, mode):
        metric = {}
        steps = 0
        # Build the model here.
        # hidden1 and hidden2 from args is to be a parameter which will be tuned
        layer1 = tf.layers.dense(data, 10, name="layer1")
        layer2 = tf.layers.dense(layer1, args.hidden1, name="layer2")
        layer3 = tf.layers.dense(layer2, args.hidden2, name="layer3")
        logits = tf.layers.dense(layer3, 1, name="logits")

        # build the graph following these case by case
        if mode == ModeKeys.INFER:
            metric['prediction'] = tf.argmax(tf.nn.sigmoid(logits), 1)
            loss = None
        else:
            loss = tf.losses.sigmoid_cross_entropy(label, logits)
            metric['loss'] = loss

        if mode == ModeKeys.TRAIN:
            global_step = tf.contrib.framework.get_global_step()
            train = tf.train.GradientDescentOptimizer(0.00001).minimize(loss, global_step)
        else:
            train = None

        if mode == ModeKeys.EVAL:
            prediction = tf.nn.sigmoid(logits)
            accuracy = tf.contrib.metrics.accuracy(tf.cast(tf.round(prediction), tf.int32), tf.cast(label, tf.int32))
            metric['accuracy'] = accuracy
            # This line is for the hyperparameter tuning
            # the cloudml always checks the summaries
            # so you just add summary like this then cloudML would see it 
            tf.summary.scalar('accuracy', accuracy)

        return metric, loss, train
    return model

# The experiment function would designed simpley like below
def experiment_fn(args):
    print("exp:", args)
    def experiment(output_dir):
        env = json.loads(os.environ.get('TF_CONFIG', '{}'))
        taskInfo = env.get('task')
        if taskInfo:
            trial = taskInfo.get('trial', '')
            if trial:
                output_dir = os.path.join(output_dir, trial)
            
        return tf.contrib.learn.Experiment(
            estimator=Estimator(
                model_fn=model_fn(args),
                model_dir=output_dir
            ),
            train_input_fn=train_input_fn(args),
            eval_input_fn=train_input_fn(args),
            train_steps=10,
            eval_steps=5
        )
    return experiment

# Add the argument "hidden1" and "hidden2" 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training arguments
    parser.add_argument(
        '--hidden1',
        type=int,
        required=True
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        required=True
    )
    parser.add_argument(
        '--output-dir',
        required=True
    )

    args = parser.parse_args()
    output_dir = args.output_dir

    learn_runner.run(experiment_fn(args), output_dir)