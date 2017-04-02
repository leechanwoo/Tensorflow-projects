"""
 * rnn tutorial
"""

import tensorflow as tf

CONSTANT = tf.app.flags
CONSTANT.DEFINE_integer("samples", 1000, "simulation data samples")
CONSTANT.DEFINE_integer("hidden", 5, "hidden layers in rnn")
CONSTANT.DEFINE_integer("vec_size", 1, "input vector size into rnn")
CONSTANT.DEFINE_integer("batch_size", 10, "minibatch size for training")
CONSTANT.DEFINE_integer("state_size", 15, "state size in rnn")
CONSTANT.DEFINE_integer("recurrent", 5, "recurrent step")
CONSTANT.DEFINE_float("learning_rate", 0.01, 'learning rate for optimizer')
CONST = CONSTANT.FLAGS

class rnn_example(object):
    """
     * rnn example module
    """

    def __init__(self):
        self._gen_sim_data()
        self._build_batch()
        self._build_model()
        self._build_train()
        self._initialize()
        # self._pack_test()

    def run(self):
        """
         * run the rnn model
        """
        self.sess.run(tf.global_variables_initializer())

        for i in range(1000):
            _, loss = self.sess.run([self.train, self.loss])
            if i % 20 == 0:
                print("loss: ", loss)

        self._close_session()

    @classmethod
    def _run_session(cls, run_graph):
        output = cls.sess.run(run_graph)
        return output

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
        cls.output, _ = tf.contrib.rnn.static_rnn(rnn_cell, tf.unstack(cls.b_train, axis=1), dtype=tf.float32)

        cls.output_w = tf.Variable(tf.truncated_normal([CONST.hidden, CONST.state_size, CONST.vec_size]))
        output_b = tf.Variable(tf.zeros([CONST.vec_size]))

        cls.pred = tf.matmul(cls.output, cls.output_w ) + output_b
        # print("output_w: ", cls.sess.run(tf.shape(cls.output_w)))
        # print("output: ", cls.sess.run(tf.shape(cls.output)))

    @classmethod
    def _build_train(cls):
        cls.loss = 0
        for i in range(CONST.hidden):
            cls.loss += tf.losses.mean_squared_error(tf.unstack(cls.b_label, axis=1)[i], cls.pred[i])
        cls.train = tf.train.AdamOptimizer(CONST.learning_rate).minimize(cls.loss)


    # @classmethod
    # def _pack_test(cls):
    #     a = tf.constant([1, 2, 3])
    #     b = tf.constant([4, 5, 6])
    #     c = tf.constant([7, 8, 9])
    #     matrix = tf.stack([a, b, c])

    #     print(cls.sess.run(matrix))
    #     print(cls.sess.run(tf.shape(matrix)))

    #     sequence = tf.unstack(matrix, axis=1)
    #     print(cls.sess.run(sequence[0]))
    #     print(cls.sess.run(sequence[1]))
    #     print(cls.sess.run(sequence[2]))

    
def main(_):
    """
     * code begins here
    """
    rnn = rnn_example()
    rnn.run()

if __name__ == "__main__":
    tf.app.run()
    