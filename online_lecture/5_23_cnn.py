
import tensorflow as tf
import os

#design CNN
col = 49
row = 61
depth = 1

image_dir = "/home/chanwoo/Sources/Temp_data_Set/DataSet/image_png/"
label_dir = "/home/chanwoo/Sources/Temp_data_Set/DataSet/label_csv/Label.csv"

image_list = [image_dir+filename for filename in os.listdir(image_dir)]
label_list = [label_dir]

image_queue = tf.train.string_input_producer(image_list)
label_queue = tf.train.string_input_producer(label_list)

img_reader = tf.WholeFileReader()
txt_reader = tf.TextLineReader()

_, image = img_reader.read(image_queue)
_, label = txt_reader.read(label_queue)

d_image = tf.image.decode_png(image)
d_data = tf.decode_csv(label, [[0]])

d_image.set_shape((row, col, depth))

x, y_ = tf.train.shuffle_batch([d_image, d_data], 100, 50000, 100)

x = tf.image.resize_images(x, (64, 64))

x = tf.cast(x, tf.float32)
label = tf.one_hot(indices=y_, depth=3, on_value=1.0, off_value=0.0, axis=1)
label = tf.reshape(label, (-1,3))
# y_ = tf.cast(y_, tf.float32)

w1 = tf.Variable(tf.truncated_normal((3, 3, 1, 16)), tf.float32)
b1 = tf.Variable(tf.zeros((100, 64, 64, 16)), tf.float32)
conv1 = tf.nn.relu(tf.nn.conv2d(x, w1, [1, 1, 1, 1], "SAME") + b1)
pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
tf.summary.image("layer1_feature", pool1[:, :, :, :1], max_outputs=2)
tf.summary.image("layer1_filter", tf.reshape(w1, (16, 3, 3, 1)), max_outputs=4)
tf.summary.histogram("hist_weight", w1)

w2 = tf.Variable(tf.truncated_normal((3, 3, 16, 32)), tf.float32)
b2 = tf.Variable(tf.zeros((100, 32, 32, 32)), tf.float32)
conv2 = tf.nn.relu(tf.nn.conv2d(pool1, w2, [1, 1, 1, 1], "SAME") + b2)
pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
tf.summary.image("layer2_feature", pool1[:, :, :, :1])
tf.summary.image("layer2_filter", tf.reshape(w2, (16, 3, 3, 32))[:, :, :, :1], max_outputs=4)
tf.summary.histogram("hist_weight2", w2)

w3 = tf.Variable(tf.truncated_normal((3, 3, 32, 64)), tf.float32)
b3 = tf.Variable(tf.zeros((100, 16, 16, 64)), tf.float32)
conv3 = tf.nn.relu(tf.nn.conv2d(pool2, w3, [1, 1, 1, 1], "SAME") + b3)
pool3 = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

flat = tf.reshape(pool3, (-1, 8*8*64))

w_fc1 = tf.Variable(tf.truncated_normal((8*8*64, 1000)), tf.float32)
b_fc1 = tf.Variable(tf.zeros((1000)), tf.float32)
fc1 = tf.nn.relu(tf.matmul(flat, w_fc1) + b_fc1)
fc1 = tf.nn.dropout(fc1, 0.7)

w_fc2 = tf.Variable(tf.truncated_normal((1000, 3)), tf.float32)
b_fc2 = tf.Variable(tf.zeros((3)), tf.float32)
fc2 = tf.matmul(fc1, w_fc2) + b_fc2

loss = tf.losses.softmax_cross_entropy(label, fc2)
train = tf.train.AdamOptimizer(1e-1).minimize(loss)

saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners( sess, coord )
    writer = tf.summary.FileWriter("./log", sess.graph)    

    summaries = tf.summary.merge_all()

    sess.run(init)
    for i in range(100):
        _, _loss, _pred, _label, _summ_str = sess.run([train, loss, tf.nn.softmax(fc2), label, summaries])
        writer.add_summary(_summ_str, i)
        if i % 10 == 0:
            print("step: ", i)
            print("loss: ", _loss)
            print("prediction: ", _pred[:10])
            print("label: ", _label[:10])

    coord.request_stop()
    coord.join(thread)