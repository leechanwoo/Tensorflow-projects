
import tensorflow as tf
import zipfile
import cProfile as p

CONST = tf.app.flags
CONST.DEFINE_integer("embed_size", 200, "embed vector size")
CONST.DEFINE_integer("vocab_size", 1000, "vocabulary size")
CONST.DEFINE_float("learning_rate", 1.0, "learning rate for optimizer")
CONST.DEFINE_float("beta1", 0.9, "beta1 for Adam optimizer")
CONST.DEFINE_float("beta2", 0.99, "beta2 for Adam optimizer")
CONST.DEFINE_float("epsilon", 1e-1, "epsilon for adam optimizer")
CONST = CONST.FLAGS

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


class skip_gram(object):
    def __init__(self):
        self._build_model()
        self._open_session()

    def run(self):
        self.sess.run(self.init)

        for i in range(100000):
            _, loss = self.sess.run([self.train, self.loss])
            
            if i % 1000 == 0:
                print(loss)

        self._close_session()

    @classmethod
    def _build_model(cls):
        words = tf.constant([i for i in range(20)])

        word_batch = tf.train.batch([words], 10, enqueue_many=True)
        label = tf.one_hot(word_batch, CONST.vocab_size, 1.0, 0.0, 1)

        embeddings = tf.Variable(tf.random_uniform((CONST.vocab_size, CONST.embed_size), -1, 1, tf.float32))

        embed = tf.nn.embedding_lookup(embeddings, word_batch)

        decode_weight = tf.Variable(tf.truncated_normal((CONST.vocab_size, CONST.embed_size)))
        decode_bias = tf.Variable(tf.zeros((CONST.vocab_size)))

        output = tf.nn.softmax(tf.matmul(embed, decode_weight, transpose_b=True) + decode_bias)

        cls.loss = tf.losses.softmax_cross_entropy(label, output)

        optimizer = tf.train.AdamOptimizer(learning_rate=CONST.learning_rate, beta1=CONST.beta1, beta2=CONST.beta2, epsilon=CONST.epsilon)
        cls.train = optimizer.minimize(cls.loss)

        cls.init = tf.global_variables_initializer()

    @classmethod
    def _open_session(cls):
        cls.sess = tf.Session()
        cls.coord = tf.train.Coordinator()
        cls.thread = tf.train.start_queue_runners(cls.sess, cls.coord)

    @classmethod
    def _close_session(cls):
        cls.coord.request_stop()
        cls.coord.join(cls.thread)
        cls.sess.close()


def my_function(words):
    counts = {}
    for w in words:
        if w in counts:
            counts[w] += 1
        else:
            counts[w] = 1
    
    # count_sorted = sorted(counts, key=lambda k: counts[k], reverse=True)
    # index = list(range(len(count_sorted)))
    # index_word = dict(zip(index, count_sorted))
    # word_index = dict(zip(count_sorted, index))
    # return index_word, word_index


def main(_):
    # word2vec = skip_gram()
    # word2vec.run()

    # words = read_data("../text8.zip")[:]
    # print("word num: ", len(words))

    # t1 = time.clock()
    # collections.Counter(words)
    # t2 = time.clock()
    # my_function(words)
    # t3 = time.clock()

    # print("collection: ", t2 - t1)
    # print("my function: ", t3 - t2)


if __name__ == "__main__":
    tf.app.run()


