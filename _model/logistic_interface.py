from _model.prepare_data import get_dataset
from _model.logistic_model import LogisticClassification
import gensim

def train_logistic(conf):
    # in_path, out_path, length, window, company_name, w2vec
    _, x_train, x_test, y_train, y_test = get_dataset(conf['logistic']['in_path'],
                                               conf['logistic']['out_path'],
                                              int(conf['logistic']['length']),
                                              int(conf['logistic']['window']),
                                               conf['logistic']['company_name'],
                                               gensim.models.Word2Vec.load(conf['w2v_model_path']))
    logistic = LogisticClassification(int(conf['embeding_size']),
                                      int(conf['logistic']['window']),
                                      int(conf['logistic']['class_num']))

    logistic.train(10, 10000, x_train, y_train, x_test, y_test, conf['logistic']['model_path'])

def test_logstic(conf):
    logistic = LogisticClassification(int(conf['embeding_size']),               
                                      int(conf['logistic']['window']),          
                                      int(conf['logistic']['class_num']))
    logistic.test(conf['logistic']['model_path'])

def logistic(conf, cmd):
    if cmd == "train":
        train_logistic(conf)
    elif cmd == "test":
        test_logstic(conf)
    else:
        print("invalid cmd: train|test")
