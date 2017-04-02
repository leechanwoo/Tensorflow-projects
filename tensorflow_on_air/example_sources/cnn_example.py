
#-*- coding: utf-8 -*-

import os
import tensorflow as tf

"""
 * tf.app은 일종의 wrapper 입니다.
 * 파이썬에서 메인 함수를 표현 할 때, __name__ == '__main__' 조건 문을 사용합니다.
 * 이 때 tf.app이라는 래퍼를 활용하면 tf.app.run() 시, 별도의 메인 함수를 찾아가는데, 그 함수가 바로
 * main(_) 함수 입니다.
 * 프로젝트를 통째로 이렇게 app으로 래핑해주고, app 클래스에서 제공하는 flags를 사용 할 수 있는데,
 * app 클래스에 상수를 등록하는 부분이라고 보시면 되겠습니다.
 * 등록하는 방법은 어렵지 않습니다.
 * tf.app.flags.DEFINE_자료형(상수이름, value, 상수에 대한 설명)
 * 이렇게 해주시면 상수이름으로 지정된 변수가 app 내부 상수로 등록됩니다.
 * C에서 말하는 #define 과 완전히 동일한 역할을 합니다.
 * 이 상수들은 tf.app.flags.FLAGS 객체의 멤버로 등록됩니다.
 * 즉, 이 상수를 참조하실 때, myconst = 6라는 상수를 하나 등록했다고 한다면,
 * value = tf.app.flags.FLAGS.myconst
 * print value --> 6 가 됩니다.
 * 이렇게 사용하시면 되겠습니다.
"""

CONST = tf.app.flags
CONST.DEFINE_string("image_dir", "DataSet/image_png/", "image directory")
CONST.DEFINE_string("label_dir", "DataSet/label_csv/Label.csv", "label directory")
CONST.DEFINE_integer("image_width", 61, "image width")
CONST.DEFINE_integer("image_height", 49, "image height")
CONST.DEFINE_float("keep_prob", 0.7, "keep probability for dropout")
CONST.DEFINE_float("learning_rate", 1e-2, "learning rate for Gradient Descent")
CONST.DEFINE_integer("epoch", 100, "epoch for learning")
CONST.DEFINE_integer("batch_size", 32, "mini mbatch size of data set")
CONST.DEFINE_integer("num_threads", 4, "number of threads for queue threading")
CONST.DEFINE_integer("capacity", 5000, "queue capacity")
CONST.DEFINE_integer("min_after_dequeue", 100, "minimum number of data when dequeue")
CONST = CONST.FLAGS


class CNN(object):
    """
    convolutional neural network class

    * CNN을 구성하고, train을 수행하는 클래스입니다.
    * 기본 기능으로 데이터 로드, 배칭, 모델 그래프 생성, 세션 및 쓰레드 구동이 제공됩니다.
    * 생성자 입력으로 이미지 파일명 리스트, 텍스트 파일명 리스트를 받으면 자동적으로 세션 및 쓰레드 구동 준비까지
    * 완료가 되며, 둘 중 하나라도 누락될 시, 수동으로 classmethod를 호출하여 구동시켜줘야 합니다.
    * 이미지 포맷은 png를 사용하고, 텍스트 파일은 csv와 호환됩니다.
    * 모델 구동 관련 함수 (run, tprint)를 제외한 모든 메서드는 protected 처리 되어 있다는 점 참고하시기 바랍니다.
    """
    def __init__(self, image_name_list=None, label_name_list=None):
        if image_name_list and label_name_list is not None:
            self._load_png(image_name_list)
            print("image ready to load")
            self._load_csv(label_name_list)
            print "csv ready to load"
            self._build_batch()
            print "batching graph created"
            self._set_variables()
            print "variables set"
            self._build_graph()
            print "op graph created"
            self._initialize()
            print "initialized, ready to run"
        else:
            print "please call the method _load_png()"
            print "please call the method _load_csv()"
            print "please call the method _build_batch()"
            print "please call the method _set_variables()"
            print "please call the method _build_graph()"
            print "please call the method _initialize()"

    def run(self):
        """
        learning the model

        * 모델을 구동하는 메서드입니다.
        * 원래 queue batch 를 사용하면 queue 내부 배치가 다 투입될 때 까지 반복문을 돌려야하지만
        * 튜토리얼이니 epoch는 사용자 지정으로 define 해주었습니다.
        * 단순히 train graph를 run 반복적으로 run 시켜주는 기능만 들어가있고, 끝나면 kill threads, close session
        * 해서 모든 프로세스를 마칩니다.
        * sess.run 아래에 스텝마다 loss와 accuracy를 보여주는 부분도 포함되어 있습니다.
        """
        for _ in range(CONST.epoch):
            self.sess.run(self.train)
            self.tprint("loss, accuracy: ", [self.out_w])

        self._close_session()

    def tprint(self, message, tensors):
        """
        print the tensor in session

        * 내가 보고 싶은 텐서의 value를 이 함수를 사용해서 볼 수 있습니다.
        * message: 사용자 임의로 출력하고 싶은 메시지를 입력합니다 ( print "hello world" 와 동일)
        * tensors: 출력하고 싶은 텐서들을 리스트 형태로 입력합니다.

        * 이 함수는 tf.Print를 이용하여 구현하였습니다.
        * Print의 첫 번째 인자는 탐색 할 그래프의 종단점을 넣어주시면 됩니다.
        * 첫 번째 그래프에서 Print의 첫 번째 인자로 들어가있는 그래프까지 경로 상에 있는 텐서만 출력이 가능하고,
        * 외부에 브랜치로 빠져있는 그래프는 탐색 영역이 아니므로 출력 시 에러를 띄웁니다.
        * 두 번째 인자에는 자신이 보고 싶은 텐서의 리스트를 넣어주면 됩니다.
        * 메시지는 위와 동일합니다.
        """
        self.sess.run(tf.Print([self.loss, self.accuracy], tensors, message))

    @classmethod
    def _initialize(cls):
        """
        * session 열기, 쓰레드 관리자 생성, 쓰레드 구동, 모든 Variables 초기화
        """
        cls.sess = tf.Session()
        cls.coord = tf.train.Coordinator()
        cls.thread = tf.train.start_queue_runners(sess=cls.sess, coord=cls.coord)
        cls.sess.run(tf.global_variables_initializer())

    @classmethod
    def _close_session(cls):
        """
        * 구동 중인 쓰레드 종료, 세션 닫기
        * 반드시 쓰레드가 먼저 모두 종료 된 후 세션을 닫아주세요.
        """
        cls.coord.request_stop()
        cls.coord.join(cls.thread)
        cls.sess.close()

    @classmethod
    def _load_png(cls, filename_list):
        """
        * WholeFileReader 클래스를 이용한 이미지 읽기.
        * qu_image_name은 이미지 파일 이름들이 들어가는 queue입니다.
        * 이 이미지들이 load 되기 시작하는 시점은 세션에서 쓰레드가 돌기 시작하는 시점입니다
        * 세션만 열어놓고 쓰레드를 돌려주지 않으면 저 부분에서 무한정 대기합니다.
        * 확장자가 다르다면 tf.image.decode_png 부분을 바꿔주시면 됩니다. (jpeg, gif, png만 가능)
        """
        ob_image_reader = tf.WholeFileReader()
        qu_image_name = tf.train.string_input_producer(filename_list)
        _, ts_image_value = ob_image_reader.read(qu_image_name)
        ts_image_decoded = tf.cast(tf.image.decode_png(ts_image_value), tf.float32)
        cls.images = tf.reshape(ts_image_decoded, [CONST.image_width, CONST.image_height, 1])

    @classmethod
    def _load_csv(cls, filename_list):
        """
        * TextLineReader 클래스를 이용한 csv 파일 읽기
        * qu_label_name은 텍스트 파일 이름이 들어가는 queue입니다.
        * 이미지 로드 함수와 동일한 queue 동작을 합니다.
        """
        ob_label_reader = tf.TextLineReader()
        qu_labelname_queue = tf.train.string_input_producer(filename_list)
        _, ts_label_value = ob_label_reader.read(qu_labelname_queue)
        ts_label_decoded = tf.cast(tf.decode_csv(ts_label_value, record_defaults=[[0]]), tf.float32)
        cls.labels = tf.cast(tf.reshape(ts_label_decoded, [1]) > 30, tf.float32)

    @classmethod
    def _build_batch(cls):
        """
        * 미니 배치를 만듭니다.
        * 이 미니 배치들이 batch queue에 저장되고, 이 queue에서 배치 한 덩어리씩 모델 학습에 투입됩니다.
        * tensors: 미니 배치로 쪼갤 풀 배치 텐서를 입력시킵니다
        * batch_size: 미니 배치의 크기 입니다
        * num_thread: queue를 구동 할 쓰레드 개수를 지정합니다
        * capacity: batch queue의 전체 크기를 정합니다
        * min_after_dequeue: queue가 비었을 때 dequeue명령이 들어오면 프로그램이 죽으므로, dequeue 했을 시,
        * 최소 잔여 데이터 크기를 설정하면 그 이하로 떨어졌을 때 dequeue 참조를 거부합니다
        * 물론 거부하고서 가만히 있진 않고 계속 파일 로드하여 queue를 다시 채워 다시 참조를 허용합니다
        """
        cls.image_batch, cls.label_batch = tf.train.shuffle_batch(
            tensors=[cls.images, cls.labels],
            batch_size=CONST.batch_size,
            num_threads=CONST.num_threads,
            capacity=CONST.capacity,
            min_after_dequeue=CONST.min_after_dequeue)

    #@classmethod
    def _set_variables(self):
        """
        * CNN에 사용될 웨이트 파라메터들입니다.
        """
        self.hidden1_w = tf.Variable(tf.truncated_normal([5, 5, 1, 32]))
        self.hidden1_b = tf.Variable(tf.zeros([32]))

        self.hidden2_w = tf.Variable(tf.truncated_normal([5, 5, 32, 64]))
        self.hidden2_b = tf.Variable(tf.truncated_normal([64]))

        self.fc_w = tf.Variable(tf.truncated_normal([49*61*64, 10]))
        self.fc_b = tf.Variable(tf.zeros([10]))

        self.out_w = tf.Variable(tf.truncated_normal([10, 1]))
        self.out_b = tf.Variable(tf.zeros([1]))

    #@classmethod
    def _build_graph(self):
        """
        * CNN모델을 graph로 구축합니다
        """
        x_image = tf.reshape(self.image_batch, [-1, CONST.image_width, CONST.image_height, 1])

        h_conv1 = tf.nn.relu(self._conv(x_image, self.hidden1_w) + self.hidden1_b)
        h_pool1 = self._max_pool(h_conv1)

        h_conv2 = tf.nn.relu(self._conv(h_pool1, self.hidden2_w) + self.hidden2_b)
        h_pool2 = self._max_pool(h_conv2)

        h_flat = tf.reshape(h_pool2, [-1, 49*61*64])

        h_fully_connected = tf.nn.relu(tf.matmul(h_flat, self.fc_w) + self.fc_b)
        h_dropout = tf.nn.dropout(h_fully_connected, CONST.keep_prob)

        self.pred = tf.nn.sigmoid(tf.matmul(h_dropout, self.out_w) + self.out_b)

        tr_entropy = tf.nn.sigmoid_cross_entropy_with_logits(self.pred, self.label_batch)
        self.loss = tf.reduce_mean(tr_entropy)
        self.train = tf.train.AdamOptimizer(CONST.learning_rate).minimize(self.loss)

        self.ev_correct_prediction = tf.equal(self.pred, self.label_batch)
        self.accuracy = tf.reduce_mean(tf.cast(self.ev_correct_prediction, tf.float32))

    @classmethod
    def _conv(cls, tensor1, tensor2):
        return tf.nn.conv2d(tensor1, tensor2, strides=[1, 1, 1, 1], padding="SAME")

    @classmethod
    def _max_pool(cls, tensor):
        return tf.nn.max_pool(tensor, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

def main(_):
    '''
    main function starting here

    * flags 주석에서 언급한 tf.app이 동작하는 main 함수입니다
    * 파이썬 전통적인 메인 함수와는 다르게 편리하게 사용 할 수 있으며, app에서 제공하는 모든 기능을 이 범위안에서
    * 자유롭게 사용 할 수 있습니다
    * 이 main 함수 내부에서 일어나는 일들은 모두 tf.app 영역이라고 보시면 됩니다.
    '''
    image_list = [CONST.image_dir+filename for filename in os.listdir(CONST.image_dir)]
    image_list.sort()
    label_list = [CONST.label_dir]

    cnn = CNN(image_list, label_list)
    cnn.run()

if __name__ == "__main__":
    """
    * tf.app을 이용한 wrapping 방법입니다
    * 단순하게 파이썬 메인 함수에다가 이 메서드 하나 호출해 주시면
    * 이 후 app에서 제공하는 기능들을 사용하면 됩니다
    * 이 소스에서는 app에서 제공하는 main 함수를 사용하여 작성했습니다.
    """
    tf.app.run()
