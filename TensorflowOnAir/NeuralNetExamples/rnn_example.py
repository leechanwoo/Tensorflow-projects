"""
* this is rnn example
"""

import tensorflow as tf

CONSTANT = tf.app.flags
CONSTANT.DEFINE_integer("epoch", 1000, "epoch when learning")
CONSTANT.DEFINE_integer("samples", 1000, "number of samples for learning")
CONSTANT.DEFINE_integer("state_size", 5, "state size in rnn ")
CONSTANT.DEFINE_integer("recurrent", 5, "number of recurrent hidden layer")
CONSTANT.DEFINE_integer("input_vector_size", 1, "input vector size")
CONSTANT.DEFINE_float("learning_rate", 0.001, "learning rate for optimizer")
CONSTANT.DEFINE_string("ckpt_dir", "./NeuralNetExamples/checkpoint/rnn.ckpt", "check point log dir")
CONSTANT.DEFINE_string("tensorboard_dir", "./NeuralNetExamples/tensorboard", "tensorboard log dir")
CONST = CONSTANT.FLAGS

class RNN(object):
    """
     * RNN model
    """
    def __init__(self):
        self._to_plot()
        self._gen_sim_data()
        self._build_batch()
        self._set_variables()
        self._build_model()
        self._save_model()
        self._build_train()
        self._initialize()

    def training(self):
        """
        * run prediction
        """
        for step in xrange(CONST.epoch):
            loss = self._run_train(step)
            if step % 100 == 0:
                self._write_checkpoint(CONST.ckpt_dir)
                print "model saved..."
                print "loss: ", loss
        print "training done"

    def prediction(self):
        """
         * run training
        """
        self._serialize_pred()

        for step in xrange(100):
            self._run_pred(step)

        self._line_plot_draw()
        self._close_session()

    @classmethod
    def _run_train(cls, step):
        cls.sess.run(tf.global_variables_initializer())
        _, loss = cls.sess.run([cls.train, cls.loss], feed_dict={cls.idx: step})
        return loss

    @classmethod
    def _run_pred(cls, step):
        cls._restore_checkpoint(CONST.ckpt_dir)
        return cls.sess.run(cls.pred, feed_dict={cls.idx: step})

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
        ts_y = tf.sin(ts_x*0.1)
        cls._line_plot("sin_x", ts_y, 1000)

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
        linear_w = tf.Variable(tf.truncated_normal([CONST.recurrent, CONST.state_size, 1]))
        linear_b = tf.Variable(tf.zeros([CONST.state_size, 1, 1]))

        cls.linear_w = tf.unpack(linear_w)
        cls.linear_b = tf.unpack(linear_b)

    @classmethod
    def _build_model(cls):
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(CONST.state_size)
        cls.input_set = tf.unpack(cls.batch_train, axis=1)
        cls.batch_label = tf.unpack(cls.batch_label, axis=1)
        cls.output, _ = tf.nn.rnn(rnn_cell, cls.input_set, dtype=tf.float32)
        cls.pred = tf.matmul(cls.output, cls.linear_w) + cls.linear_b

    @classmethod
    def _serialize_pred(cls):
        cls.pred_sin = tf.reshape(cls.pred, (100,))
        cls._line_plot("pred_sin", cls.pred_sin, 100)


    @classmethod
    def _build_train(cls):
        cls.loss = 0
        for i in xrange(CONST.state_size):
            cls.loss += cls._mean_square_error(cls.pred[i], cls.batch_label[i])

        cls.train = tf.train.AdamOptimizer(CONST.learning_rate).minimize(cls.loss)

    @classmethod
    def _mean_square_error(cls, batch, label):
        return tf.reduce_mean(tf.pow(batch - label, 2))

    @classmethod
    def _line_plot(cls, name, tensor, length):
        cls.length = length
        cls.plot = tensor[cls.idx]
        tf.summary.scalar(name, cls.plot)

    @classmethod
    def _line_plot_draw(cls):
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(CONST.tensorboard_dir)
        for i in xrange(cls.length):
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
    rnn = RNN()
    rnn.prediction()
    print "end process"

if __name__ == "__main__":
    tf.app.run()
    