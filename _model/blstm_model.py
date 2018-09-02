import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pandas as pd

class BLSTM_WSD(object):

    def __init__(self, max_seq_length=30, embedding_size=100, hidden_units=50):
        range = int(max_seq_length/2)
        self.x_input = tf.placeholder(dtype=tf.float32,
                                      shape=[None, max_seq_length, embedding_size],
                                      name='x_input')

        self.y_input = tf.placeholder(dtype=tf.float32,
                                      shape=[None, 2],
                                      name='y_input')

        # rnn layer
        pre_x_input = tf.slice(self.x_input, begin=[0, 0, 0], size=[-1, range, -1])
        pre_rnn_output = self.rnn_layer(pre_x_input, hidden_units, "preceding_lstm")
        sec_x_input = tf.slice(self.x_input, begin=[0, range, 0], size=[-1, -1, -1])
        sec_rnn_output = self.rnn_layer(sec_x_input, hidden_units, "succeeding_lstm")

        # concat
        rnn_output = tf.concat([pre_rnn_output, sec_rnn_output], 1)

        self.y_output = self.fully_connect_layer(rnn_output, hidden_units*2, 2)
        self.loss = self.loss_function(self.y_input, self.y_output)

        #self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
        self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def rnn_layer(self, input_x, hidden_units, name):
        def length(sequence):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
            length = tf.reduce_sum(used, 1)
            length = tf.cast(length, tf.int32)
            return length

        with tf.variable_scope(name):
            cell = rnn.BasicLSTMCell(num_units=hidden_units, state_is_tuple=True)
            cell_drop = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1, output_keep_prob=1)
            outputs, last_states = tf.nn.dynamic_rnn(cell=cell_drop,
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

    def count_neg(self, y_output, y_input):
        count_neg = 0
        count_recall = 0
        total = len(y_input)
        for y_out, y_in in zip(y_output, y_input):
            if y_in[0] == 0:
                count_neg += 1
                if y_out[0] < 0.5:
                    count_recall += 1
        recall = 0
        if count_neg > 0:
            recall = count_recall/count_neg
        return total, count_neg, recall

    def count_pos(self, y_output, y_input):
        count_pos = 0
        count_recall = 0
        total = len(y_input)
        for y_out, y_in in zip(y_output, y_input):
            if y_in[0] == 1:
                count_pos += 1
                if y_out[0] > 0.5:
                    count_recall += 1
        recall = 0
        if count_pos > 0:
            recall = count_recall/count_pos
        return total, count_pos, recall

    def bad_case(self, y_output, y_input, x_info):
        goodcase = {"company":[], "real":[], "predict":[], "sentence": [], "feature word list":[]}
        badcase = {"company":[], "real":[], "predict":[], "sentence": [], "feature word list":[]}
        for y_out, y_in, info in zip(y_output, y_input, x_info):
            if np.argmax(y_out, 0) == np.argmax(y_in, 0):
                columns = ['company', 'real', 'predict', 'sentence', 'feature word list']
                goodcase["company"].append(info[0])
                goodcase["real"].append("%.3f"%(y_in[0]))
                goodcase["predict"].append("%.3f"%(y_out[0]))
                goodcase["sentence"].append(info[1])
                goodcase["feature word list"].append(str(info[2]))
                #print("good case: [%s]"%info[0])
                #print('y_true: (%.3f, %.3f), y_out: (%.3f, %.3f)'%(y_in[0], y_in[1], y_out[0], y_out[1]))
                #print("primary sentence:")
                #print(info[1])
                #print('feature word list:')
                #print(info[2])
                #print('--------------------------------------------------')
                continue
            print("Bad case: [%s]"%info[0])
            print('y_true: %.3f, y_out: %.3f'%(y_in[0], y_out[0]))
            print("primary sentence:")
            print(info[1])
            print('feature word list:')
            print(info[2])
            print('--------------------------------------------------')
            columns = ['company', 'real', 'predict', 'sentence', 'feature word list']
            badcase["company"].append(info[0])
            badcase["real"].append("%.3f"%(y_in[0]))
            badcase["predict"].append("%.3f"%(y_out[0]))
            badcase["sentence"].append(info[1])
            badcase["feature word list"].append(str(info[2]))
        df_bad = pd.DataFrame(badcase)
        df_bad.to_excel("/home/op/work/survey/log/lstm_eval_badcase.xlsx", index=False, columns=columns)
        df_good = pd.DataFrame(goodcase)
        df_good.to_excel("/home/op/work/survey/log/lstm_eval_goodcase.xlsx", index=False, columns=columns)

    def train_and_test(self, x_train, y_train, x_test, y_test, x_test_info, epoch, batch_size, path):
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
                    accu = self.accuracy(y_out, y_input)
                    print('[epoch:%d] [batch_num:%d] loss=%9f accu=%.3f' % (i, j, loss, accu))

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

            self.bad_case(y_output, y_test, x_test_info)

            saver = tf.train.Saver()
            self.save(sess, saver, path, 0)

    def save(self, sess, saver, path, step):
        saver.save(sess, path + 'model.ckpt', step)

    def load(self, sess, saver, path):
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise NotADirectoryError

    def predict(self, sess, x_input, y_input, x_info):
        feed_dict = {self.x_input:x_input, self.y_input:y_input}
        y_output = sess.run(self.y_output, feed_dict=feed_dict)

        accu = self.accuracy(y_output, y_input)
        total, pos, recall_pos = self.count_pos(y_output, y_input)
        _, neg, recall_neg = self.count_neg(y_output, y_input)

        self.bad_case(y_output, y_input, x_info)
        print("accu:  %.3f,  recall_pos: %.3f,  recall_neg: %.3f"%(accu, recall_pos, recall_neg))
        print("total case: %d, positive: %d, negtive: %d"%(total, pos, neg))

    def evaluate(self, x_input, y_input, x_info, model_path):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.load(sess, saver, model_path)
            self.predict(sess, x_input, y_input, x_info)
