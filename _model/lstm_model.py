import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class Text_LSTM(object):
    def __init__(self, max_seq_length=70, embedding_size=100, hidden_units=32):

        self.x_input = tf.placeholder(dtype=tf.float32,
                                      shape=[None, max_seq_length, embedding_size],
                                      name='x_input')

        self.y_input = tf.placeholder(dtype=tf.float32,
                                      shape=[None, 2],
                                      name='y_input')

        rnn_output = self.rnn_layer(self.x_input, hidden_units)
        self.y_output = self.fully_connect_layer(rnn_output, hidden_units, 2)
        #self.y_output = rnn_output
        self.loss = self.loss_function(self.y_input, self.y_output)

        #self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)

    def rnn_layer(self, input_x, hidden_units):
        def length(sequence):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
            length = tf.reduce_sum(used, 1)
            length = tf.cast(length, tf.int32)
            return length

        cell = rnn.BasicLSTMCell(num_units=hidden_units, state_is_tuple=True)
        outputs, last_states = tf.nn.dynamic_rnn(cell=cell,
                                                 dtype=tf.float32,
                                                 sequence_length = length(input_x),
                                                 inputs=input_x)
        output = last_states.h
        return output  #返回最后一个状态  LSTMStateTuple.h

    def fully_connect_layer(self, input_tensor, hidden_units, class_num):
        self.w = tf.Variable(tf.random_normal(shape=[hidden_units, class_num], stddev=0.01),
                             name="weight")
        self.b = tf.Variable(tf.zeros(shape=[class_num]),
                                      name='bias')
        y = tf.nn.softmax(tf.matmul(input_tensor, self.w)+self.b, name='softmax')
        return y

    def loss_function(self, y_real, y_output):
        #loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_output, labels=y_real)
        loss = -tf.reduce_mean(y_real * tf.log(y_output))
        return loss

    def accuracy(self, y_output, y_input):
        y_pred = np.equal(np.argmax(y_output, axis=1), np.argmax(y_input, axis=1))
        accu = np.mean(y_pred.astype(np.float32))
        return accu

    def train_and_test(self, x_train, y_train, x_test, y_test, epoch, batch_size):
        train_sample_num = len(y_train)
        batch_num = (int)(train_sample_num/batch_size)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(epoch):
                for j in range(batch_num):
                    x_input = x_train[j*batch_size : (j+1)*batch_size]
                    y_input = y_train[j*batch_size : (j+1)*batch_size]
                    feed_dict = {self.x_input:x_input, self.y_input:y_input}
                    _, loss, y_out = sess.run((self.train_op, self.loss, self.y_output), feed_dict=feed_dict)
                    print('[epoch:%d] [batch_num:%d] loss=%9f' % (i, j, loss))
                    #print(y_out)
                    #print(y_input)

                #test
                feed_dict = {self.x_input: x_test, self.y_input: y_test}
                y_output = sess.run(self.y_output, feed_dict=feed_dict)
                accu = self.accuracy(y_output, y_test)
                print('accuracy: %f'%accu)

                #shuffle
                np.random.seed(i)
                shuffle_indices = np.random.permutation(np.arange(len(y_train)))
                x_train = x_train[shuffle_indices]
                y_train = y_train[shuffle_indices]

    def prediction(self, x_input, y_input):
        with tf.Session() as sess:
            feed_dict = {self.x_input:x_input, self.y_input:y_input}
            y_output = sess.run(self.y_output, feed_dict=feed_dict)
            accu = self.accuracy(y_output, y_input)
        return accu

