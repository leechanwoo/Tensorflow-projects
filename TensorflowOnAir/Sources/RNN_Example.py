
import tensorflow as tf

class RNN(object):
    """
     * RNN model
    """
    def __init__(self):
        self._gen_sim_data()
        self._build_batch()
        self._set_variables()
        self._build_graph()
        self._initialize()

    def run(self):
        """
         * run the rnn model
        """
        self._run()

    @classmethod
    def _run(cls):
        """
         * run the rnn model
        """
        for step in xrange(1000):
            _, _loss = cls.sess.run([cls.train, cls.loss])
            if step % 100 == 0:
                print _loss

        print "process done"
        cls._close_session()

    @classmethod
    def _gen_sim_data(cls):
        ts_x = tf.constant(range(0, 1001, 1), dtype=tf.float32)
        ts_y = tf.sin(ts_x*0.1)
        cls.ts_batch_y = tf.reshape(ts_y[:-1], [200, 5, 1])
        cls.ts_batch_y_ = tf.reshape(ts_y[1:], [200, 5, 1])

    @classmethod
    def _build_batch(cls):
        cls.batch_train, cls.batch_label = tf.train.batch(
            [cls.ts_batch_y, cls.ts_batch_y_],
            20,
            enqueue_many=True)

    @classmethod
    def _initialize(cls):
        cls.sess = tf.Session()
        cls.sess.run(tf.global_variables_initializer())
        cls.coord = tf.train.Coordinator()
        cls.thread = tf.train.start_queue_runners(cls.sess, cls.coord)


    @classmethod
    def _close_session(cls):
        cls.coord.request_stop()
        cls.coord.join(cls.thread)
        cls.sess.close()


    @classmethod
    def _build_graph(cls):
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(5)
        cls.input_set = tf.unpack(cls.batch_train, axis=1)
        cls.batch_label = tf.unpack(cls.batch_label, axis=1)
        cls.output, _ = tf.nn.rnn(rnn_cell, cls.input_set, dtype=tf.float32)
        cls.pred = tf.matmul(cls.output, cls.linear_w) + cls.linear_b
        
        cls.loss = 0
        for i in xrange(5):        
            cls.loss += tf.reduce_mean(tf.pow(cls.pred[i] - cls.batch_label[i], 2))
        
        cls.train = tf.train.AdamOptimizer(0.001).minimize(cls.loss)

    @classmethod
    def _set_variables(cls):
        cls.linear_w = tf.unpack(tf.Variable(tf.truncated_normal([5, 5, 1])))
        cls.linear_b = tf.unpack(tf.Variable(tf.zeros([5, 1, 1])))

def main(_):
    """
    * code start here
    """
    rnn = RNN()
    rnn.run()


if __name__ == "__main__":
    tf.app.run()
