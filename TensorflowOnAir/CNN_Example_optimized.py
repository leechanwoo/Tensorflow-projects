
import os
#import tensorflow as tf
from tensorflow import image, nn, train, app, Session
from tensorflow import Variable, global_variables_initializer
from tensorflow import WholeFileReader, TextLineReader, decode_csv
from tensorflow import cast, reshape, matmul, float32
from tensorflow import reduce_mean, argmax, truncated_normal, zeros, equal

CONST = app.flags
CONST.DEFINE_string("image_dir", "../Temp_data_Set/Test_Dataset_png/", "image directory")
CONST.DEFINE_string("label_dir", "../Temp_data_Set/Test_Dataset_csv/Label.csv", "label directory")
CONST.DEFINE_integer("image_width", 61, "image width")
CONST.DEFINE_integer("image_height", 49, "image height")
CONST.DEFINE_float("keep_prob", 0.7, "keep probability for dropout")
CONST.DEFINE_float("learning_rate", 1e-4, "learning rate for Gradient Descent")
CONST.DEFINE_integer("epoch", 100, "epoch for learning")
CONST.DEFINE_integer("batch_size", 32, "mini mbatch size of data set")
CONST.DEFINE_integer("num_thread", 4, "number of threads for queue threading")
CONST.DEFINE_integer("capacity", 5000, "queue capacity")
CONST.DEFINE_integer("min_after_dequeue", 100, "minimum number of data when dequeue")
CONST = CONST.FLAGS


class CNN(object):
    """
    convolutional neural network class
    """
    def __init__(self, image_name_list, label_name_list):
        self._load_png(image_name_list)
        self._load_csv(label_name_list)
        self._build_batch()
        self._build_graph()
        print "ready to run"

    def run(self):
        """
        run the model
        """
        with Session() as sess:
            coord = train.Coordinator()
            thread = train.start_queue_runners(sess=sess, coord=coord)

            sess.run(global_variables_initializer())

            for _ in range(CONST.epoch):
                sess.run(self.train)
                print "loss: ", sess.run(self.loss)
                print "accuracy: ", sess.run(self.accuracy)

        coord.request_stop()
        coord.join(thread)

    @classmethod
    def _load_png(cls, filename_list):
        """
        load png images
        """
        ob_image_reader = WholeFileReader()
        qu_image_name = train.string_input_producer(filename_list)
        _, ts_image_value = ob_image_reader.read(qu_image_name)
        ts_image_decoded = cast(image.decode_png(ts_image_value), float32)
        cls.images = reshape(ts_image_decoded, [CONST.image_width, CONST.image_height, 1])

    @classmethod
    def _load_csv(cls, filename_list):
        """
        load csv file
        """
        ob_label_reader = TextLineReader()
        qu_labelname_queue = train.string_input_producer(filename_list)
        _, ts_label_value = ob_label_reader.read(qu_labelname_queue)
        ts_label_decoded = cast(decode_csv(ts_label_value, record_defaults=[[0]]), float32)
        cls.labels = reshape(ts_label_decoded, [1])

    @classmethod
    def _build_batch(cls):
        """
        building mini batches
        """
        cls.image_batch, cls.label_batch = train.shuffle_batch(
            tensors=[cls.images, cls.labels],
            batch_size=CONST.batch_size,
            num_threads=CONST.num_threads,
            capacity=CONST.capacity,
            min_after_dequeue=CONST.min_after_dequeue)

    @classmethod
    def _build_graph(cls):
        """
        building graph
        """
        cls._set_variables()

        x_image = reshape(cls.image_batch, [-1, CONST.image_width, CONST.image_height, 1])

        h_conv1 = nn.relu(cls._conv(x_image, cls.hidden1_w) + cls.hidden1_b)
        h_pool1 = cls._max_pool(h_conv1)

        h_conv2 = nn.relu(cls._conv(h_pool1, cls.hidden2_w) + cls.hidden2_b)
        h_pool2 = cls._max_pool(h_conv2)

        h_flat = reshape(h_pool2, [-1, 49*61*64])

        h_fully_connected = nn.relu(matmul(h_flat, cls.fc_w) + cls.fc_b)
        h_dropout = nn.dropout(h_fully_connected, CONST.keep_prob)

        cls.pred = matmul(h_dropout, cls.out_w) + cls.out_b

        tr_entropy = nn.sigmoid_cross_entropy_with_logits(cls.pred, cls.label_batch)
        cls.loss = reduce_mean(tr_entropy)
        cls.train = train.AdamOptimizer(CONST.learning_rate).minimize(cls.loss)

        ev_correct_prediction = equal(argmax(cls.pred, 1), argmax(cls.label_batch, 1))
        cls.accuracy = reduce_mean(cast(ev_correct_prediction, float32))

    @classmethod
    def _conv(cls, tensor1, tensor2):
        return nn.conv2d(tensor1, tensor2, strides=[1, 1, 1, 1], padding="SAME")

    @classmethod
    def _max_pool(cls, tensor):
        return nn.max_pool(tensor, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

    @classmethod
    def _set_variables(cls):
        cls.hidden1_w = Variable(truncated_normal([5, 5, 1, 32]))
        cls.hidden1_b = Variable(zeros([32]))

        cls.hidden2_w = Variable(truncated_normal([5, 5, 32, 64]))
        cls.hidden2_b = Variable(truncated_normal([64]))

        cls.fc_w = Variable(truncated_normal([49*61*64, 10]))
        cls.fc_b = Variable(zeros([10]))

        cls.out_w = Variable(truncated_normal([10, 1]))
        cls.out_b = Variable(zeros([1]))


def main(_):
    '''
    main function starting here
    '''
    image_list = os.listdir(CONST.image_dir)

    for i in xrange(len(image_list)):
        image_list[i] = CONST.image_dir + image_list[i]

    label_list = [CONST.label_dir]

    cnn = CNN(image_list, label_list)
    cnn.run()

if __name__ == "__main__":
    app.run()
