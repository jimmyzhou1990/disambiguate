import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pandas as pd
import random

class BLSTM_WSD(object):

    def __init__(self, max_seq_length=30, embedding_size=100, batch_size=128,  hidden_units=50, word_keep_prob=1.0, w2vec=None, model_name='model', attention=False):
        self.range = int(max_seq_length/2)
        self.w2vec = w2vec
        self.drop_vec = w2vec['UnknownWord']
        self.word_keep_prob = word_keep_prob
        self.model_name = model_name
        self.gate = 0.5
        self.batch_size = batch_size
        self.attention = attention

        self.x_input = tf.placeholder(dtype=tf.float32,
                                      shape=[None, max_seq_length, embedding_size],
                                      name='x_input')

        self.y_input = tf.placeholder(dtype=tf.float32,
                                      shape=[None, 2],
                                      name='y_input')

        # rnn layer
        with tf.name_scope("slice"):
            pre_x_input = tf.slice(self.x_input, begin=[0, 0, 0], size=[-1, self.range, -1], name='slice_for_preceding')
            sec_x_input = tf.slice(self.x_input, begin=[0, self.range, 0], size=[-1, -1, -1], name='slice_for_secceding')
        pre_rnn_output = self.rnn_layer(pre_x_input, hidden_units, "pre_lstm", 'static')
        sec_rnn_output = self.rnn_layer(sec_x_input, hidden_units, "suc_lstm", 'static')

        # concat
        with tf.name_scope("concat_rnn"):
            rnn_concat = tf.concat([pre_rnn_output, sec_rnn_output], 1)

        # hidden layer
        rnn_hidden_out = hidden_units*2
        with tf.name_scope("hidden_layer"):
            rnn_output = self.rnn_output_hidden_layer(rnn_concat, rnn_hidden_out)

        # softmax
        with tf.name_scope("soft_max"):
            self.y_output = self.softmax_layer(rnn_output, rnn_hidden_out, 2)

        # loss
        with tf.name_scope("loss_fun"):
            self.loss = self.loss_function(self.y_input, self.y_output)

        # accuracy
        with tf.name_scope("accuracy"):
            self.accuracy = self.accuracy()

        #self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
        with tf.name_scope("Optimizer"):
            self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def drop_word(self, x_input):
        def seq_length(seq):
            used = np.sign(np.max(np.abs(seq), 2))
            length = np.sum(used, 1, dtype=np.int32)
            return length

        word_keep_prob = self.word_keep_prob
        #print(seq_length(x_input))
        for index, seq_len in enumerate(seq_length(x_input)):
            drop_num = (int)((1 - word_keep_prob)*seq_len) if seq_len >= 10 else 0
            #print("word_keep_prob: %f, seq_len: %d, drop_num: %d"%(word_keep_prob, seq_len, drop_num))
            if drop_num == 0:
                return x_input
            drop_indexs = random.sample(range(seq_len), drop_num)
            #print("drop_index: %s"%drop_indexs)
            for i in drop_indexs:
                x_input[index][i] = self.drop_vec
            return x_input
    
    def rnn_output_hidden_layer(self, input_x, out_size):
        in_size = input_x.get_shape().as_list()[1]
        self.w = tf.Variable(tf.random_normal(shape=[in_size, out_size], stddev=0.01),
                             name="weight")
        self.b = tf.Variable(tf.zeros(shape=[out_size]),
                                      name='bias')
        y = tf.matmul(input_x, self.w)+self.b
        return y

    def rnn_layer(self, input_x, hidden_units, name, type='dynamic'):
        def length(sequence):
            if type == 'dynamic':
                used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
                length = tf.reduce_sum(used, 1)
                length = tf.cast(length, tf.int32)
            else:
                used = tf.sign(tf.reduce_max(tf.abs(sequence+1), 2))
                length = tf.reduce_sum(used, 1)
                length = tf.cast(length, tf.int32)
            return length

        with tf.variable_scope(name):
            sequence_length = length(input_x)
            cell = rnn.BasicLSTMCell(num_units=hidden_units, state_is_tuple=True)
            cell_drop = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1, output_keep_prob=1)
            #cell_atten = tf.contrib.seq2seq.AttentionWrapper(cell_drop, 3, state_is_tuple=True)
            outputs, last_states = tf.nn.dynamic_rnn(cell=cell_drop,
                                                     dtype=tf.float32,
                                                     sequence_length = sequence_length,
                                                     inputs=input_x)
            if self.attention:
                output = self.attention_layer(outputs, hidden_units, hidden_units, name)
            else:
                output = last_states.h
            # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
            #output = self.attention_layer(outputs, hidden_units, hidden_units, name)
            return output  #返回最后一个状态  LSTMStateTuple.h

    def attention_layer(self, rnn_outputs, rnn_units, attention_units, name):
        steps = rnn_outputs.get_shape().as_list()[1]
        attention_w = tf.Variable(tf.random_normal(shape=[rnn_units, attention_units], stddev=0.1),
                                  name=name+"_attention_w")
        attention_b = tf.Variable(tf.random_normal(shape=[attention_units], stddev=0.1),
                                  name=name+"_attention_b")
        
        attn_u = tf.tanh(tf.matmul(tf.reshape(rnn_outputs, [-1, rnn_units]), attention_w) + attention_b)

        u_w = tf.Variable(tf.random_normal(shape=[attention_units, 1], stddev=0.1),
                                           name=name+"_attention_uw")
        
        attn_z = tf.reshape(tf.matmul(attn_u, u_w), [-1, steps])  # [batch_size*step, 1]  => [batch_size, steps]
        attn_p = tf.nn.softmax(attn_z) #[batch_size, steps]  
        attn_ps = tf.stack([attn_p]*rnn_units, axis=2)
        output = tf.reduce_sum(tf.multiply(rnn_outputs, attn_ps), 1)
        
        if name == 'pre_lstm':
            self.pre_attn = attn_p
        else:
            self.sec_attn = attn_p

        return output

    def softmax_layer(self, input_tensor, hidden_units, class_num):
        self.w = tf.Variable(tf.random_normal(shape=[hidden_units, class_num], stddev=0.01),
                             name="weight")
        self.b = tf.Variable(tf.zeros(shape=[class_num]),
                                      name='bias')
        y = tf.nn.softmax(tf.matmul(input_tensor, self.w)+self.b, name='softmax')
        return y

    def loss_function(self, y_real, y_output):
        #loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_output, labels=y_real)
        loss = -tf.reduce_mean(y_real * tf.log(tf.clip_by_value(y_output, 1e-8, 1.0)))
        return loss

    def accuracy(self, y_output, y_input):
        correct_predictions = tf.equal(tf.argmax(y_output, axis=1), tf.argmax(y_input, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        return accuracy

    def bad_case(self, y_output, y_input, x_info, attn=None,  to_excel=False, print_goodcase=False):
        goodcase = {"company":[], "real":[], "predict":[], "sentence": [], "feature word list":[]}
        badcase = {"company":[], "real":[], "predict":[], "sentence": [], "feature word list":[]}
        for index, (y_out, y_in, info) in enumerate(zip(y_output, y_input, x_info)):
            if attn:
                attn_p = attn[0][index]
                attn_s = attn[1][index]
            y_out_p = y_out[0]

            y_out[0] = 1 if y_out[0] >= self.gate else 0
            if len(info[2]) == 0 and len(info[3]) == 0:
                print('y_true: %.3f, y_out: %.3f'%(y_in[0], y_out_p))
                print("primary sentence:")
                print(info[1])
                print('feature word list:')
                print(info[2])
                print('--------------------------------------------------')

            if np.argmax(y_out, 0) == np.argmax(y_in, 0):
                columns = ['company', 'real', 'predict', 'sentence', 'feature word list']
                goodcase["company"].append(info[0])
                goodcase["real"].append("%.3f"%(y_in[0]))
                goodcase["predict"].append("%.3f"%(y_out_p))
                goodcase["sentence"].append(info[1])
                goodcase["feature word list"].append(str(info[2]+info[3]))
                if print_goodcase:
                    print("good case: [%s]"%info[0])
                    print('y_true: %.3f, y_out: %.3f'%(y_in[0], y_out_p))
                    print("primary sentence:")
                    print(info[1])
                    print('feature word list:')
                    if attn:
                        print('preceding:')
                        print(list(zip(info[2], attn_p)))
                        print(attn_p[len(info[2]):])
                        print('secceding:')
                        print(list(zip(info[3], attn_s)))
                        print(attn_s[len(info[3]):])
                    else:
                        print('preceding:')
                        print(info[2])
                        print('secceding:')
                        print(info[3])
                    print('--------------------------------------------------')
                continue
            print("******!!!******Bad case: [%s]"%info[0])
            print('y_true: %.3f, y_out: %.3f'%(y_in[0], y_out_p))
            print("primary sentence:")
            print(info[1])
            print('feature word list:')
            if attn:
                print('preceding:')
                print(list(zip(info[2], attn_p)))
                print(attn_p[len(info[2]):])
                print('secceding:')
                print(list(zip(info[3], attn_s)))
                print(attn_s[len(info[3]):])
            else:
                print('preceding:')
                print(info[2])
                print('secceding:')
                print(info[3])
            print('--------------------------------------------------')
            columns = ['company', 'real', 'predict', 'sentence', 'feature word list']
            badcase["company"].append(info[0])
            badcase["real"].append("%.3f"%(y_in[0]))
            badcase["predict"].append("%.3f"%(y_out_p))
            badcase["sentence"].append(info[1])
            badcase["feature word list"].append(str(info[2]+info[3]))
        if to_excel:
            df_bad = pd.DataFrame(badcase)
            df_bad.to_excel("/home/op/work/survey/log/lstm_eval_badcase_%s.xlsx"%self.model_name, index=False, columns=columns)
            df_good = pd.DataFrame(goodcase)
            df_good.to_excel("/home/op/work/survey/log/lstm_eval_goodcase_%s.xlsx"%self.model_name, index=False, columns=columns)

    def train_and_test(self, x_train, y_train, x_test, y_test, x_test_info, epoch, path):
        train_sample_num = len(y_train)
        batch_size = self.batch_size
        batch_num = (int)(train_sample_num/batch_size)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            writer = tf.summary.FileWriter(path+'/summary', sess.graph)
            tf.summary.scalar('test_accuracy', self.accuracy)
            merged = tf.summary.merge_all()
            for i in range(epoch):
                for j in range(batch_num):
                    x_input = x_train[j*batch_size : (j+1)*batch_size]
                    x_input = self.drop_word(x_input)
                    y_input = y_train[j*batch_size : (j+1)*batch_size]
                    feed_dict = {self.x_input:x_input, self.y_input:y_input}
                    _, loss, accuracy = sess.run((self.train_op, self.loss, self.accuracy), feed_dict=feed_dict)
                    print('[epoch:%d] [batch_num:%d] loss=%9f accu=%.3f' % (i, j, loss, accuracy))

                #test
                feed_dict = {self.x_input: x_test, self.y_input: y_test}
                accuracy, y_output, summary = sess.run((self.accuracy, self.y_output, merged), feed_dict=feed_dict)
                writer.add_summary(summary, i)
                print('test accuracy: %f' % accuracy)

                #shuffle
                np.random.seed(i)
                shuffle_indices = np.random.permutation(np.arange(len(y_train)))
                x_train = x_train[shuffle_indices]
                y_train = y_train[shuffle_indices]

            self.bad_case(y_output, y_test, x_test_info)

            saver = tf.train.Saver()
            self.save(sess, saver, path, 0)
        #writer.close()

    def save(self, sess, saver, path, step):
        saver.save(sess, path + self.model_name + '/' + self.model_name + '.ckpt', step)

    def load(self, sess, saver, path):
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise NotADirectoryError

    def count_neg(self, y_output, y_input):
        count_neg = 0
        count_recall = 0
        total = len(y_input)
        for y_out, y_in in zip(y_output, y_input):
            if y_in[0] == 0:
                count_neg += 1
                if y_out[0] < self.gate:
                    count_recall += 1
        recall = 0
        if count_neg > 0:
            recall = count_recall / count_neg
        return total, count_neg, count_recall, recall

    def count_pos(self, y_output, y_input):
        count_pos = 0
        count_recall = 0
        total = len(y_input)
        for y_out, y_in in zip(y_output, y_input):
            if y_in[0] == 1:
                count_pos += 1
                if y_out[0] >= self.gate:
                    count_recall += 1
        recall = 0
        if count_pos > 0:
            recall = count_recall / count_pos
        return total, count_pos, count_recall, recall


    def predict(self, sess, x_input, y_input, x_info):
        feed_dict = {self.x_input: x_input, self.y_input: y_input}
        attn = None
        if self.attention:
            y_output, pre_attn, sec_atten = sess.run((self.y_output, self.pre_attn, self.sec_attn), feed_dict=feed_dict)
            attn = (pre_attn.tolist(), sec_atten.tolist())
        else:
            y_output = sess.run(self.y_output, feed_dict=feed_dict)

        accu = self.accuracy(y_output, y_input)
        total, pos, rpos, recall_pos = self.count_pos(y_output, y_input)
        _, neg, rneg, recall_neg = self.count_neg(y_output, y_input)

        self.bad_case(y_output, y_input, x_info, attn=attn,  to_excel=True, print_goodcase=False)
        print("accu:  %.3f,  recall_pos: %.3f,  recall_neg: %.3f" % (accu, recall_pos, recall_neg))
        print("total case: %d, positive: %d, negtive: %d" % (total, pos, neg))
        return pos, rpos, neg, rneg

    def evaluate(self, x_input, y_input, x_info, model_path, model_name, gate):
        self.model_name = model_name
        self.gate = gate
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.load(sess, saver, model_path)
            pos, rpos, neg, rneg = self.predict(sess, x_input, y_input, x_info)
            return pos, rpos, neg, rneg
