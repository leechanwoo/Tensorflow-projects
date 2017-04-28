
import os
import tensorflow as tf
import matplotlib.image as im

# import csv
# file_dir = "/home/chanwoo/Sources/Temp_data_Set/DataSet/label_csv/_Label.csv"
# save_dir = "/home/chanwoo/Sources/Temp_data_Set/DataSet/label_csv/Label.csv"

# data_list = []
# with open(file_dir, 'r') as f:
#     data = csv.reader(f)
#     for i in data:
#         data_list.append([int(i[0])])

# with open(save_dir, 'w') as f:
#     writer = csv.writer(f, delimiter=',')
#     for d in data_list:
#         if d[0] < 20:  
#             writer.writerow('0')
#         elif d[0] < 40:    
#             writer.writerow('1')
#         else: 
#             writer.writerow('2')

image_dir = "/home/chanwoo/Sources/Temp_data_Set/DataSet/image_png/"
image_width = 49
image_height = 61

image_list = [image_dir + i for i in os.listdir(image_dir)]

label_dir = "/home/chanwoo/Sources/Temp_data_Set/DataSet/label_csv/Label.csv"
label_list = [label_dir]

imagename_queue = tf.train.string_input_producer(image_list)
labelname_queue = tf.train.string_input_producer(label_list)

image_reader = tf.WholeFileReader()
label_reader = tf.TextLineReader()

image_key, image_value = image_reader.read(imagename_queue)
label_key, label_value = label_reader.read(labelname_queue)

image_decoded = tf.cast(tf.image.decode_png(image_value, channels=1), tf.float32)
image_decoded.set_shape([image_height, image_width, 1])

label_decoded = tf.decode_csv(label_value, record_defaults=[[0]])

x, y_= tf.train.shuffle_batch(tensors=[image_decoded, label_decoded], batch_size=128, num_threads=4, capacity=5000, min_after_dequeue=100)
y_ = tf.one_hot(indices=y_, depth=3, on_value=1.0, off_value=0.0, axis=1, dtype=tf.float32)
y_ = tf.reduce_mean(y_, axis=2)

# parameters
hidden1_w = tf.Variable(tf.truncated_normal([5,5,1,10]))
hidden1_b = tf.Variable(tf.zeros([10]))

hidden2_w = tf.Variable(tf.truncated_normal([5,5,10,10]))
hidden2_b = tf.Variable(tf.zeros([10]))

fc_w = tf.Variable(tf.truncated_normal([49*61*10, 10]))
fc_b = tf.Variable(tf.zeros([10]))

out_w = tf.Variable(tf.truncated_normal([10, 3]))
out_b = tf.Variable(tf.zeros([3]))

# CNN Model
h_conv1 = tf.nn.relu(tf.nn.conv2d(x, hidden1_w, strides=[1, 1, 1, 1], padding="SAME") + hidden1_b)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, hidden2_w, strides=[1, 1, 1, 1], padding="SAME") + hidden2_b)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

h_flat = tf.reshape(h_pool2, [-1, 49*61*10])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, fc_w) + fc_b)

drop_fc = tf.nn.dropout(h_fc1, 0.7)

y = tf.sigmoid(tf.matmul(drop_fc, out_w) + out_b)

loss = tf.losses.softmax_cross_entropy(y_, y)
train = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        _, _loss, _accuracy = sess.run([train, loss, accuracy])
        print("run the session")
        print ("loss: ", _loss)
        print ("accuracy: ", _accuracy)
    
    coord.request_stop()
    coord.join(thread)
