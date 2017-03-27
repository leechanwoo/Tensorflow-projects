"""
 * rnn tutorial
"""

import tensorflow as tf
import matplotlib.pyplot as plt

CONSTANT = tf.app.flags
CONSTANT.DEFINE_integer("samples", 1000, "simulation data samples")
CONSTANT.DEFINE_integer("hidden", 5, "hidden layers in rnn")
CONSTANT.DEFINE_integer("vec_size", 1, "input vector size into rnn")
CONSTANT.DEFINE_integer("batch_size", 10, "minibatch size for training")
CONSTANT.DEFINE_integer("state_size", 15, "state size in rnn")
CONSTANT.DEFINE_integer("recurrent", 5, "recurrent step")
CONST = CONSTANT.FLAGS

class rnn_example(object):
    """
     * rnn example module
    """

    def __init__(self):
        self._gen_sim_data()
        self._build_batch()
        self._build_model()
        self._initialize()

    def run(self):
        """
         * run the rnn model
        """
        print("batch input shape")
        print(self._run_session(tf.shape(self.b_train)))
        print("label input shape")
        print(self._run_session(tf.shape(self.b_label)))
        self._close_session()

    @classmethod
    def _run_session(cls, run_graph):
        output = cls.sess.run(run_graph)
        return output

    @classmethod
    def _initialize(cls):
        cls.sess = tf.Session()

    @classmethod
    def _close_session(cls):
        cls.sess.close()

    @classmethod
    def _gen_sim_data(cls):
        cls.ts_x = tf.constant([i for i in range(CONST.samples+1)], dtype=tf.float32)
        ts_y = tf.sin(cls.ts_x * 0.1)

        sp_batch = (int(CONST.samples/CONST.hidden), CONST.hidden, CONST.vec_size)
        cls.batch_input = tf.reshape(ts_y[:-1], sp_batch)
        cls.batch_label = tf.reshape(ts_y[1:], sp_batch)

    @classmethod
    def _build_batch(cls):
        batch_set = [cls.batch_input, cls.batch_label]
        cls.b_train, cls.b_label = tf.train.batch(batch_set, CONST.batch_size, enqueue_many=True)

    @classmethod
    def _build_model(cls):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(CONST.state_size)
        output, _ = tf.nn.dynamic_rnn(rnn_cell, tf.unstack(cls.b_train, axis=1), dtype=tf.float32)

def main(_):
    """
     * code begins here
    """
    # rnn = rnn_example()
    # rnn.run()

    plt.plot([1, 2, 3, 4])
    plt.show()

if __name__ == "__main__":
    tf.app.run()
    