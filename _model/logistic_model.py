import tensorflow as tf
import numpy as np
import scipy.stats as stats

class LogisticClassification(object):

    def __init__(self, embedding_size, window, class_num):
        self.x_input = tf.placeholder(dtype=tf.float32,
                                      shape=[None, window*2, embedding_size],
                                      name='x_input')
        self.y_input = tf.placeholder(dtype=tf.float32,
                                      shape=[None, class_num],
                                      name='y_input')

        self.w = tf.Variable(tf.random_normal(shape=[embedding_size, class_num], stddev=0.01),
                             name="weight")
        self.b = tf.Variable(tf.zeros(shape=[class_num]),
                             name='bias')

        nor = stats.norm(0, window/1.96)
        norm_weight = list(nor.pdf(range(window+1)))
        norm_weight_reverse = norm_weight.copy()
        norm_weight_reverse.reverse()
        wc = np.reshape(np.array(norm_weight[1:] + norm_weight_reverse[:-1], dtype='float32'),
                [window*2, 1])
        print(wc)
        
        x_trans = tf.transpose(self.x_input, [0, 2, 1])
        x_reshape = tf.reshape(x_trans, [-1, window*2])
        wx = tf.reshape(tf.matmul(x_reshape, wc), [-1, embedding_size])

        activation = tf.matmul(wx, self.w) # + self.b
        self.y_output = tf.nn.softmax(activation, name='softmax')
        self.loss = self.loss_function(self.y_output, self.y_input)
        self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

    def loss_function(self, y_output, y_real):
        cross_entropy = -tf.reduce_mean(y_real * (tf.log(tf.clip_by_value(y_output, 1e-8, 1))))
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_output, labels=y_real)
        return cross_entropy

    def train(self, epoch, batch_size, x_train, y_train, x_test, y_test, model_path):
        train_sample_num = len(y_train)
        batch_num = (int)(train_sample_num/batch_size)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(epoch):
                for j in range(batch_num):
                    x_input = x_train[j*batch_num : (j+1)*batch_num]
                    y_input = y_train[j*batch_num : (j+1)*batch_num]
                    feed_dict = {self.x_input: x_input, self.y_input: y_input}
                    _, loss, y_out = sess.run((self.train_op, self.loss, self.y_output), feed_dict=feed_dict)
                    print('[epoch:%d] [batch_num:%d] loss=%9f' % (i, j, loss))
                #test
                feed_dict = {self.x_input: x_test, self.y_input: y_test}
                y_output = sess.run(self.y_output, feed_dict=feed_dict)
                print(y_output)
                accu = self.accuracy(y_output, y_test)
                print(accu)

                #shuffle
                np.random.seed(i)
                shuffle_indices = np.random.permutation(np.arange(len(y_train)))
                x_train = x_train[shuffle_indices]
                y_train = y_train[shuffle_indices]
            print(sess.run(self.w)) 
            saver = tf.train.Saver()
            self.save(sess, saver, model_path, 0)
            
    def accuracy(self, y_output, y_input):
        y_pred = np.equal(np.argmax(y_output, axis=1), np.argmax(y_input, axis=1))
        accu = np.mean(y_pred.astype(np.float32))
        return accu

    def save(self, sess, saver, path, step):
        saver.save(sess, path + 'model.ckpt', step)

    def load(self, sess, saver, path):
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

    def test(self, path, x_input=None, y_input=None):
        saver = tf.train.Saver()
        with tf.Session() as sess: 
            self.load(sess, saver, path)
            print("model w and b:")
            print(sess.run(self.w))
            print(sess.run(self.b))
            feed_dict = {self.x_input: x_input, self.y_input: y_input}
            y_output = sess.run(self.y_output, feed_dict=feed_dict)
            print(y_output)
            accu = self.accuracy(y_output, y_input)
            print(accu)





