
import tensorflow as tf
import matplotlib.pyplot as plt

data = []
label = []
for i in range(1000):
    if i % 2 == 0:
        data.append([float(i-5) * 0.1 for i in range(10)])
        label.append([1])
    else:
        data.append([float(5-i) * 0.1 for i in range(10)])
        label.append([0])

x = tf.constant(data, tf.float32)
y_ = tf.constant(label, tf.float32)

with tf.name_scope("layer_1"):
    w1 = tf.Variable(tf.truncated_normal(shape=(10,20)), name="weight_1")
    b1 = tf.Variable(tf.zeros(shape=(20)), name="bias_1")
    layer1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
    tf.summary.histogram("wegiht1", w1)
    tf.summary.histogram("bias1", b1)

with tf.name_scope("layer_2"):
    w2 = tf.Variable(tf.truncated_normal(shape=(20,10)), name="weight_2")
    b2 = tf.Variable(tf.zeros(shape=(10)), name="bias_2")
    layer2 = tf.nn.sigmoid(tf.matmul(layer1, w2) + b2)
    tf.summary.histogram("weight2", w2)
    tf.summary.histogram("bias2", b2)

with tf.name_scope("layer_out"):
    wo = tf.Variable(tf.truncated_normal(shape=(10,1)), name="wegith_out")
    bo = tf.Variable(tf.zeros(shape=(1)), name="bias_out")
    pred = tf.nn.sigmoid(tf.matmul(layer2, wo) + bo)
    tf.summary.histogram("weight_out", wo)
    tf.summary.histogram("bias_out", bo)

with tf.name_scope("loss_function"):
    loss = tf.losses.sigmoid_cross_entropy(y_, pred)
    train = tf.train.GradientDescentOptimizer(1e-1).minimize(loss)
    tf.summary.scalar("loss", loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./online_lecture/log", sess.graph)
    for i in range(3000):
        _, _loss, _summ = sess.run([train, loss, merged])
        writer.add_summary(_summ, i)
        if i % 100 == 0:
            print("step: %d, loss: %f"%(i, _loss))

    test = sess.run(pred)

    print(data[:10])
    print(test[:10])
    print()