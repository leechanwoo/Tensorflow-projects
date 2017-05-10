
import tensorflow as tf

data = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]

label = [1, 0, 1, 0]

x = tf.placeholder(dtype=tf.float32)
y_ = tf.placeholder(dtype=tf.float32)
feed_dict = {x : data, y_ : label}

y_ = tf.reshape(y_, (4, 1))

layer_w = tf.Variable(tf.truncated_normal(shape=(10,1)))  #input x output 개수
layer_b = tf.Variable(tf.truncated_normal(shape=(1,1)))   # output (=y) 개수랑 동일 
y = tf.nn.sigmoid(tf.matmul(x,layer_w)+layer_b)

cost_function = -y_*tf.log(y) - (1-y_)*tf.log(1-y)



with tf.Session() as sess :
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    a, b, c, d,f,y_,e = sess.run([x,y,layer_w,layer_b,y,y_,cost_function],feed_dict)
    print("a is",a)
    print("b is",b)
    print("c is",c)
    print("d is",d)
    print("f is",f)
    print("y_ is",y_)
    print("e is",e)




