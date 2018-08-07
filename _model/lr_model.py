from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

class LR_Model(object):

    def __init__(self):
        self.lr = LogisticRegression(solver='saga')

    def train(self, x_train, y_train):
        self.lr.fit(x_train, y_train)

    def test(self, x_test, y_test, x_test_info):
        #score = self.lr.score(x_test, y_test)
        #print('Score: %f'%score)
        result = self.lr.predict_proba(x_test)
        print(result[0:10])
        print(y_test[0:10])

        r = [0 if r[1]<0.5 else 1 for r in result]
        
        num = len(y_test)
        acu_count=0
        rec_count=0
        pos_count = np.sum(y_test)
        
        for index, (y, r, y_out) in enumerate(zip(y_test, r, result)):
            if y == 1. and r == 1.:
                rec_count +=1
            if y == r:
                acu_count += 1
            if  y != r:
                print("Bad case,   y_ture:%.3f,   y_out:(%.3f, %.3f)"%(y, y_out[0], y_out[1]))
                print("company name:%s"%x_test_info[index][0])
                print("primary sentence:")
                print(x_test_info[index][1])
                print("similarity and distance:")
                print(x_test_info[index][2])
                print("topn list:")
                print("similarity        distance          word")
                for top in x_test_info[index][3]:
                    print("%.5f           %.5f           %s"%(top[0], top[1], top[2]))
                print("------------------------------------------------------")



        print("recall: %f"%(rec_count/pos_count))
        print("accu: %f"%(acu_count/num))


class SVM_Model(object):
    def __init__(self):
        self.svm = SVC()

    def train(self, x_train, y_train):
        self.svm.fit(x_train, y_train)

    def test(self, x_test, y_test):
        result = self.svm.predict(x_test)
        print(result[0:10])
        print(y_test[0:10])

        r = [0 if r<0.3 else 1 for r in result]

        count=0
        for y, r in zip(y_test, r):
            if y == r:
                count += 1

        print("score: %f"%(count/len(y_test)))





