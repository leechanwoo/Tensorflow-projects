
import os
#from PIL import Image
import tensorflow as tf

image_dir = "/home/chanwoo/Sources/Temp_data_Set/DataSet/image_jpeg"
image_width = 194
image_height = 146

image_list = os.listdir(image_dir)

for i in range(len(image_list)):
    image_list[i] = image_dir + image_list[i]

label_dir = "/home/chanwoo/Sources/Temp_data_Set/DataSet/label_csv/Label.csv"
label_list = [label_dir]

imagename_queue = tf.train.string_input_producer(image_list)
labelname_queue = tf.train.string_input_producer(label_list)

image_reader = tf.WholeFileReader()
label_reader = tf.TextLineReader()

image_key, image_value = image_reader.read(imagename_queue)
label_key, label_value = label_reader.read(labelname_queue)

image_decoded = tf.cast(tf.image.decode_jpeg(image_value, channels=3), tf.float32)
label_decoded = tf.cast(tf.decode_csv(label_value, record_defaults=[[0]]), tf.float32)

label = tf.reshape(label_decoded, [1])
image = tf.reshape(image_decoded, [image_width, image_height, 3])

x, y_= tf.train.shuffle_batch(tensors=[image, label], batch_size=1, num_threads=4, capacity=5000, min_after_dequeue=100)

# parameters
hidden1_w = tf.Variable(tf.truncated_normal([5,5,1,32]))
hidden1_b = tf.Variable(tf.zeros([32]))

hidden2_w = tf.Variable(tf.truncated_normal([5,5,32,64]))
hidden2_b = tf.Variable(tf.truncated_normal([64]))

fc_w = tf.Variable(tf.truncated_normal([49*61*64, 10]))
fc_b = tf.Variable(tf.zeros([10]))

out_w = tf.Variable(tf.truncated_normal([10, 1]))
out_b = tf.Variable(tf.zeros([1]))

# CNN Model
x_image = tf.reshape(x, [-1, image_width, image_height, 1])

h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, hidden1_w, strides=[1, 1, 1, 1], padding="SAME") + hidden1_b)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, hidden2_w, strides=[1, 1, 1, 1], padding="SAME") + hidden2_b)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

h_flat = tf.reshape(h_pool2, [-1, 49*61*64])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, fc_w) + fc_b)

drop_fc = tf.nn.dropout(h_fc1, 0.7)

pred = tf.matmul(drop_fc, out_w) + out_b

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y_))
train = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        print ("loss: ", sess.run(loss))
        print ("accuracy: ", sess.run(accuracy))
    
    coord.request_stop()
    coord.join(thread)
