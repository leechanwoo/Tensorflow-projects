import tensorflow as tf

CONST = tf.app.flags
CONST.DEFINE_integer("epoch", 1000, "epoch when learning")
CONST.DEFINE_integer("samples", 1000, "number of samples for learning")
CONST.DEFINE_integer("state_size", 5, "state size in rnn ")
CONST.DEFINE_integer("recurrent", 5, "number of recurrent hidden layer")
CONST.DEFINE_integer("input_vector_size", 1, "input vector size")
CONST.DEFINE_float("learning_rate", 0.001, "learning rate for optimizer")
CONST = CONST.FLAGS

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
        for step in xrange(CONST.epoch):
            loss = self._run()
            if step % 100 == 0:
                print loss
        print "training done"
        self._close_session()

    @classmethod
    def _run(cls):
        _, loss = cls.sess.run([cls.train, cls.loss])
        return loss

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
    def _gen_sim_data(cls):
        ts_x = tf.constant(range(CONST.samples+1), dtype=tf.float32)
        ts_y = tf.sin(ts_x*0.1)
        
        sz_batch = (CONST.samples/CONST.state_size, CONST.state_size, CONST.input_vector_size)
        cls.ts_batch_y = tf.reshape(ts_y[:-1], sz_batch)
        cls.ts_batch_y_ = tf.reshape(ts_y[1:], sz_batch)

    @classmethod
    def _build_batch(cls):
        cls.batch_train, cls.batch_label = tf.train.batch(
            [cls.ts_batch_y, cls.ts_batch_y_],
            20,
            enqueue_many=True)

    @classmethod
    def _set_variables(cls):
        cls.linear_w = tf.Variable(tf.truncated_normal([CONST.recurrent, CONST.state_size, 1]))
        cls.linear_b = tf.Variable(tf.zeros([CONST.state_size, 1, 1]))

        cls.linear_w = tf.unpack(cls.linear_w)
        cls.linear_b = tf.unpack(cls.linear_b)

    @classmethod
    def _build_graph(cls):
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(CONST.state_size)
        cls.input_set = tf.unpack(cls.batch_train, axis=1)
        cls.batch_label = tf.unpack(cls.batch_label, axis=1)
        cls.output, _ = tf.nn.rnn(rnn_cell, cls.input_set, dtype=tf.float32)
        cls.pred = tf.matmul(cls.output, cls.linear_w) + cls.linear_b

        cls.loss = 0
        for i in xrange(CONST.state_size):
            cls.loss += cls._mean_square_error(cls.pred[i], cls.batch_label[i])

        cls.train = tf.train.AdamOptimizer(CONST.learning_rate).minimize(cls.loss)

    @classmethod
    def _mean_square_error(cls, batch, label):
        return tf.reduce_mean(tf.pow(batch - label, 2))

    @classmethod
    def _line_plot(cls, tensor, length):
        
        pass
    
    @classmethod
    def _line_plot_draw(cls):
        
        pass

def main(_):
    """
    * code start here
    """
    rnn = RNN()
    rnn.run()

if __name__ == "__main__":
    tf.app.run()