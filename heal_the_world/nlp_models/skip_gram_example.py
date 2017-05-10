"""
* this is the skip gram example
"""

import tensorflow as tf

CONSTANT = tf.app.flags
CONSTANT.DEFINE_integer("dict_size", 50000, "dictionary size")
CONSTANT.DEFINE_integer("embed_size", 200, "embeded dictionary size")
CONSTANT.DEFINE_integer("num_sampled", 64, "num sampled for nce loss function")
CONSTANT.DEFINE_integer("learning_rate", 1e-4, "learning rate")
CONSTANT.DEFINE_integer("epoch", 100, "epoch for training")
CONST = CONSTANT.FLAGS


class SkipGram(object):
    """
     * skip gram class
    """
    def __init__(self):
        self._build_skipgram()
        self._initialization()

    def training(self):
        """
         * run the train
        """

        for step in xrange(CONST.epoch):
            loss = self._run_train()
            if step % 10 == 0:
                print loss

        self._close_session()

    def prediction(self):
        """
         * run the prediction
        """
        embeddings = self._run_prediction()
        print embeddings[0]
        self._close_session()

    @classmethod
    def _run_train(cls):
        _, loss = cls.sess.run([cls.optimizer, cls.loss])
        return loss

    @classmethod
    def _run_prediction(cls):
        embeddings = cls._get_embeddings()
        return cls.sess.run(embeddings)

    @classmethod
    def _gen_onehots(cls):
        indices = tf.cast(tf.transpose([range(CONST.dict_size), range(CONST.dict_size)]), tf.int64)
        values = tf.ones((CONST.dict_size), dtype=tf.int64)
        shape = (CONST.dict_size, CONST.dict_size)
        st_onehots = tf.SparseTensor(indices=indices, values=values, shape=shape)
        cls.ts_onehots = tf.sparse_tensor_to_dense(st_onehots)

    @classmethod
    def _set_variables(cls):
        cls.embed_dict = tf.Variable(tf.random_uniform((CONST.dict_size, CONST.embed_size)))

    @classmethod
    def _build_skipgram(cls):
        sz_weights = (CONST.dict_size, CONST.embed_size)

        init_embeddings = tf.random_uniform(sz_weights, -1, 1)
        cls.embeddings = tf.Variable(init_embeddings)

        stddev = 1.0/tf.sqrt(tf.cast(CONST.embed_size, tf.float32))
        init_nce_weight = tf.truncated_normal(sz_weights, stddev=stddev)
        w_nce = tf.Variable(init_nce_weight)

        init_nce_bias = tf.zeros((CONST.dict_size))
        nce_bias = tf.Variable(init_nce_bias)

        x_word = tf.constant([1, 2, 3, 4, 5])
        y_word = tf.constant([[4], [5], [6], [7], [8]])
        embeds = tf.nn.embedding_lookup(cls.embeddings, x_word)

        cls.loss = tf.nn.nce_loss(
            weights=w_nce,
            biases=nce_bias,
            inputs=embeds,
            labels=y_word,
            num_sampled=CONST.num_sampled,
            num_classes=CONST.dict_size
        )

        cls.optimizer = tf.train.GradientDescentOptimizer(CONST.learning_rate).minimize(cls.loss)

    @classmethod
    def _get_embeddings(cls):
        norm = tf.sqrt(tf.reduce_sum(tf.square(cls.embeddings), 1, keep_dims=True))
        return cls.embeddings/norm

    @classmethod
    def _initialization(cls):
        cls.sess = tf.Session()
        cls.sess.run(tf.global_variables_initializer())

    @classmethod
    def _close_session(cls):
        cls.sess.close()

def main(_):
    """
    * this is main function of this project
    """
    skip_gram = SkipGram()
    #skip_gram.training()
    skip_gram.prediction()
    print "process done"

if __name__ == "__main__":
    tf.app.run()
