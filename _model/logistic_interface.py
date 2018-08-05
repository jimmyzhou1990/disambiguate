from _model.prepare_data import get_dataset
from _model.logistic_model import LogisticClassification
import gensim

def train_logistic(conf):
    # in_path, out_path, length, window, company_name, w2vec
    _, x_train, x_test, y_train, y_test = get_dataset(conf['logistic']['in_path'],
                                               conf['logistic']['out_path'],
                                               conf['logistic']['length'],
                                               conf['logistic']['window'],
                                               conf['COMPANY_POS'],
                                               conf['w2v_model_path'])
    logistic = LogisticClassification(int(conf['embeding_size']),
                                      int(conf['logistic']['window']),
                                      int(conf['logistic']['class_num']))

    logistic.train(10, 100, x_train, y_train, x_test, y_test)
