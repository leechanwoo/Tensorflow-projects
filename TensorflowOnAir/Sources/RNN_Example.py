
import tensorflow as tf

class RNN(object):
    """
     * RNN model
    """
    def __init__(self):
        self._gen_sim_data()
        self._build_batch()
        self._initialize()

    def run(self):
        """
         * run the rnn model
        """
        probabilities = []
        loss = 0

        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(5)
        print self.sess.run(self.batch_train)
        output, state = tf.nn.rnn(rnn_cell, self.batch_train)
        

        logits = tf.matmul(output, self.softmax_w) + self.softmax_b
        probabilities.append(tf.nn.softmax(logits))
        loss += tf.nn.softmax_cross_entropy_with_logits(probabilities, self.batch_label )
        
        train = tf.train.AdamOptimizer(0.001).minimize(loss)

        for _ in xrange(2):
            _, _loss = self.sess.run([train, loss])
            print _loss


        self._close_session()

    @classmethod
    def _gen_sim_data(cls):
        ts_x = tf.constant(range(0, 1001, 1), dtype=tf.float32)
        cls.ts_y = tf.reshape(tf.sin(ts_x*0.1), [1001, 1])

    @classmethod
    def _build_batch(cls):
        cls.batch_train, cls.batch_label = tf.train.batch([cls.ts_y[:-1], cls.ts_y[1:]], 5, enqueue_many=True)

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
        pass

    @classmethod
    def _set_variables(cls):
        cls.softmax_w = tf.Variable(tf.truncated_normal([5,1]))
        cls.softmax_b = tf.Variable(tf.zeros([1]))

def main(_):
    """
    * code start here
    """
    rnn = RNN()
    rnn.run()


if __name__ == "__main__":
    tf.app.run()
