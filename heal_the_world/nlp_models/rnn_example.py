"""
* this is rnn example
"""

import tensorflow as tf

CONSTANT = tf.app.flags
CONSTANT.DEFINE_integer("epoch", 100, "epoch when learning")
CONSTANT.DEFINE_integer("samples", 3000000, "number of samples for learning")
CONSTANT.DEFINE_integer("state_size", 10, "state size in rnn ")
CONSTANT.DEFINE_integer("recurrent", 10, "number of recurrent hidden layer")
CONSTANT.DEFINE_integer("input_vector_size", 300, "input vector size")
CONSTANT.DEFINE_float("learning_rate", 0.001, "learning rate for optimizer")
CONSTANT.DEFINE_string("ckpt_dir", "./checkpoint/rnn.ckpt", "check point log dir")
CONSTANT.DEFINE_string("tensorboard_dir", "./tensorboard", "tensorboard log dir")
CONSTANT.DEFINE_integer("batch_size", 100, "mini batch size")
CONSTANT.DEFINE_bool("is_training_mode", False, "select training or prediction")
CONST = CONSTANT.FLAGS

class RNN(object):
    """
     * RNN model
    """
    def __init__(self):
        self._to_plot()
        print "ready to visualize"
        self._gen_sim_data()
        print "generated data"
        self._build_batch()
        print "batch built"
        self._set_variables()
        print "variables set"
        self._build_model()
        print "model built"
        self._save_model()
        print "saver created"
        self._build_train()
        print "loss graph created"
        self._initialize()
        print "initialized"

    def training(self):
        """
        * run prediction
        """
        print "start training...."
        for step in xrange(CONST.epoch):
            loss = self._run_train()
            if step % 10 == 0:
                print "step: ", step
                print "loss: ", loss

        print "model saving..."
        self._write_checkpoint(CONST.ckpt_dir)
        print "training done"

    def prediction(self):
        """
         * run training
        """
        self._run_pred()
        self._line_plot_draw(600)
        self._close_session()

    @classmethod
    def _run_train(cls):
        cls.sess.run(tf.global_variables_initializer())
        _, loss = cls.sess.run([cls.train, cls.loss])
        return loss

    @classmethod
    def _run_pred(cls):
        cls._restore_checkpoint(CONST.ckpt_dir)
        return cls.sess.run(cls.pred)

    @classmethod
    def _save_model(cls):
        cls.saver = tf.train.Saver()

    @classmethod
    def _write_checkpoint(cls, directory):
        cls.saver.save(cls.sess, directory)

    @classmethod
    def _restore_checkpoint(cls, directory):
        cls.saver.restore(cls.sess, directory)

    @classmethod
    def _initialize(cls):
        cls.sess = tf.Session()
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
        ts_y = tf.sin(ts_x*0.01)
        # print "ts_y"
        #cls._print(ts_y[1:21])

        sz_batch = (
            CONST.samples/(CONST.recurrent*CONST.input_vector_size),
            CONST.recurrent,
            CONST.input_vector_size)

        cls.ts_batch_y = tf.reshape(ts_y[:-1], sz_batch)
        cls.ts_batch_y_ = tf.reshape(ts_y[1:], sz_batch)

    @classmethod
    def _build_batch(cls):
        batch_set = [cls.ts_batch_y, cls.ts_batch_y_]
        cls.b_train, cls.b_label = tf.train.batch(batch_set, CONST.batch_size, enqueue_many=True)

    @classmethod
    def _set_variables(cls):
        sz_weight = (CONST.recurrent, CONST.state_size, CONST.input_vector_size)
        sz_bias = (CONST.recurrent, 1, CONST.input_vector_size)
        linear_w = tf.Variable(tf.truncated_normal(sz_weight))
        linear_b = tf.Variable(tf.zeros(sz_bias))

        cls.linear_w = tf.unpack(linear_w)
        cls.linear_b = tf.unpack(linear_b)

    @classmethod
    def _build_model(cls):
        rnn_cell = tf.nn.rnn_cell.LSTMCell(CONST.state_size)
        cls.input_set = tf.unpack(cls.b_train, axis=1)
        cls.label_set = tf.unpack(cls.b_label, axis=1)
        cls.output, _ = tf.nn.rnn(rnn_cell, cls.input_set, dtype=tf.float32)
        cls.pred = tf.matmul(cls.output, cls.linear_w) + cls.linear_b

        if CONST.is_training_mode is not True:
            cls._line_plot("1_label_set", tf.transpose(cls.label_set, (1, 0, 2)))
            cls._line_plot("2_pred_sin", tf.transpose(cls.pred, (1, 0, 2)))

    @classmethod
    def _build_train(cls):
        cls.loss = 0
        for i in xrange(CONST.recurrent):
            cls.loss += cls._mean_square_error(cls.pred[i], cls.label_set[i])

        cls.train = tf.train.AdamOptimizer(CONST.learning_rate).minimize(cls.loss)

    @classmethod
    def _mean_square_error(cls, batch, label):
        return tf.reduce_mean(tf.pow(batch - label, 2))

    @classmethod
    def _line_plot(cls, name, tensor):
        ts_vector = tf.reshape(
            tf.pack(tensor),
            (CONST.batch_size*CONST.recurrent*CONST.input_vector_size,))
        # print "ts_vector"
        # cls._print(ts_vector[:20])
        plot = ts_vector[cls.idx]
        tf.summary.scalar(name, plot)

    @classmethod
    def _print(cls, tensor):
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            thread = tf.train.start_queue_runners(sess, coord)
            sess.run(tf.global_variables_initializer())
            print sess.run(tensor)
            coord.request_stop()
            coord.join(thread)

    @classmethod
    def _line_plot_draw(cls, num_plot):
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(CONST.tensorboard_dir)
        for i in xrange(num_plot):
            summary_str = cls.sess.run(summaries, {cls.idx: i})
            writer.add_summary(summary_str, i)
            writer.flush()
        writer.close()

    @classmethod
    def _to_plot(cls):
        cls.idx = tf.placeholder(tf.int32)

def main(_):
    """
    * code start here
    """
    print "code start"
    rnn = RNN()
    if CONST.is_training_mode is True:
        rnn.training()
    else:
        rnn.prediction()
    print "end process"

if __name__ == "__main__":
    tf.app.run()
    