import tensorflow as tf
import numpy as np
import csv


tf.logging.set_verbosity(tf.logging.INFO)


def gen_dataset():
    """ data generator """
    up = [i for i in range(10)]
    down = [9-i for i in range(10)]

    with open("./test.csv", 'w') as f:
        writer = csv.writer(f, delimiter=",")
        for i in range(5):
            writer.writerow([1] + up)
            writer.writerow([0] + down)


def input_fn():
    """ input function for estimator """
    dataset = tf.contrib.data.TextLineDataset("./test.csv")
    dataset.shuffle(2000)
    dataset = dataset.batch(5)

    itr = dataset.make_one_shot_iterator()
    batch = itr.get_next()

    data = tf.decode_csv(batch, [[0]]*11)

    train = tf.stack(data[1:], axis=1)
    label = tf.expand_dims(data[0], axis=1)

    train = tf.cast(train, tf.float32)
    label = tf.cast(label, tf.float32)
    return train, label


def model_fn(features, labels, mode):
    """ model function for estimator """
    layer1 = tf.layers.dense(features, 10)
    layer2 = tf.layers.dense(layer1, 10)
    layer3 = tf.layers.dense(layer2, 10)
    layer4 = tf.layers.dense(layer3, 10)
    layer5 = tf.layers.dense(layer4, 10)
    out = tf.layers.dense(layer5, 1)


    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        loss = tf.losses.sigmoid_cross_entropy(labels, out)
        optimizer = tf.train.GradientDescentOptimizer(1e-2)
        train_op = optimizer.minimize(loss, global_step)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            train_op=train_op,
            loss=loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.sigmoid_cross_entropy(labels, out)
        acc = tf.metrics.accuracy(labels, tf.round(tf.nn.sigmoid(out)))
        eval_metric_ops = {"acc": acc}
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops)


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'prob': tf.nn.sigmoid(out)})




est = tf.estimator.Estimator(model_fn, model_dir='./est')

for epoch in range(10):
    est.train(input_fn)
    est.evaluate(input_fn)




