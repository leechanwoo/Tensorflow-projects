
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

train_image, train_label = tf.train.shuffle_batch([d_image, d_data], 100)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners( sess, coord )


    W = tf.Variable()
    b = tf.Variable()
    tf.nn.conv2d()

    conv1 = tf.layers.conv2d(train_image, 10, 3, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

    conv2 = tf.layers.conv2d(pool1, 10, 3, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

    conv3 = tf.layers.conv2d(pool2, 10, 3, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer)
    conv4 = tf.layers.conv2d(conv3, 10, 3, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer)
    conv5 = tf.layers.conv2d(conv4, 10, 3, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer)
    conv6 = tf.layers.conv2d(conv4, 10, 3, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer)
    conv7 = tf.layers.conv2d(conv3, 10, 3, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer)
    conv8 = tf.layers.conv2d(conv3, 10, 3, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer)

    _images, _labels = sess.run([tf.shape(train_image), tf.shape(train_label)])
        
    coord.request_stop()
    coord.join(thread)

exit()

#batching 
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

train_image, train_label = tf.train.shuffle_batch([d_image, d_data], 100)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners( sess, coord )

    for i in range(100):
        _images, _labels = sess.run([tf.shape(train_image), tf.shape(train_label)])
        print("images: ", _images)
        print("labels: ", _labels)
        print()
        
    coord.request_stop()
    coord.join(thread)



# reading text
label_dir = "/home/chanwoo/Sources/Temp_data_Set/DataSet/label_csv/test_text.csv"
filename_list = [label_dir]

filename_queue = tf.train.string_input_producer(filename_list)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

#col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = tf.decode_csv(value, [[0] for i in range(10)])
data = tf.decode_csv(value, [[0] for i in range(10)])

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess, coord)

    for i in range(10):
        _data = sess.run(data)
        print(_data)

    coord.request_stop()
    coord.join(thread)


# reading images
image_dir = "/home/chanwoo/Sources/Temp_data_Set/DataSet/image_png/"
filename_list = [image_dir+filename for filename in os.listdir(image_dir)]

filename_queue = tf.train.string_input_producer(filename_list)

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

image = tf.image.decode_jpeg(value)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners( sess, coord )

    sess.run(image)    

    coord.request_stop()
    coord.join(thread)
