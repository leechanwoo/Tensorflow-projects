
import tensorflow as tf

output_size = 10
input_size = 10
state_size = 5
time_step = 7


init_state = tf.Variable([[0]*state_size], trainable=False, dtype=tf.float32)

input_data = [
    tf.Variable([list(range(input_size))], trainable=False, dtype=tf.float32) \
    for _ in range(time_step)
    ]
    
label = [
    tf.Variable([list(range(1,output_size+1))], trainable=False, dtype=tf.float32) \
    for _ in range(time_step)
    ]


def build_rnn():
    init_val = lambda x,y: tf.truncated_normal(shape=(x,y))
    init_zeros = lambda x: tf.zeros(shape=(x,), dtype=tf.float32)

    Wi = tf.Variable(init_val(input_size, state_size))
    Ws = tf.Variable(init_val(state_size, state_size))
    bs = tf.Variable(init_zeros(state_size))
    Wo = [
        tf.Variable(init_val(state_size, output_size)) \
        for _ in range(time_step)
        ]

    state = init_state
    out = []
    pred = []
    states = []
    for i in range(time_step):
        state = tf.matmul(input_data[i], Wi) + tf.matmul(state, Ws) + bs
        states.append(state)
        out.append(tf.nn.tanh(state))
        pred.append(tf.matmul(out[i], Wo[i]))

    return out, pred, states


def train_model():
    loss = 0
    for i in range(time_step):
        loss += tf.losses.mean_squared_error(label[i], pred[i])

    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    return train, loss

out, pred, states = build_rnn()
train, loss = train_model()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    writer = tf.summary.FileWriter("./CheckPoint", sess.graph)
    for i in range(1000):
        _, _loss = sess.run([train, loss])
        print("loss: ", _loss)

    print("predictions")
    for j in range(time_step):
        print(sess.run(pred[j]))
