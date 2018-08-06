from sklearn.linear_model import LogisticRegression

class LR_Model(object):

    def __init__(self):
        self.lr = LogisticRegression()

    def train(self, x_train, y_train):
        self.lr.fit(x_train, y_train)

    def test(self, x_test, y_test):
        score = self.lr.score(x_test, y_test)
        print('Score: %f'%score)