import sys
from _corpus.company_corpus import CorpusFactory
from _corpus.clean_corpus import CorpusCleaner
from _model.word2vector import w2vectorFactory
from _disambiguate.disambiguate import Disambiguate
from _model.logistic_interface import logistic
from _model.lr_model import LR_Model
from _model.lr_model import SVM_Model
from _model.prepare_data import get_lr_model_dataset

def load_config(conf):
    with open("/home/op/work/survey/data/company_disambiguate.txt") as f:
        for line in f:
            company = {}
            company['full_name'], company['short_name'] = line.strip().split('\t')[0:2]
            conf['company_list'].append(company)

conf = {
    'company_list' : [
        # {
        #     'full_name'  :  '星星集团有限公司',
        #     'short_name' :  '星星'
        # },
    ],
    'corpus_path'    : '/home/op/work/survey/corpus',
    'user_dict'      : '/home/op/work/survey/data/user_dict.txt',
    'stopwords_path' : '/home/op/work/survey/data/stop_word.txt',
    'w2v_model_path'     : '/home/op/work/survey/model/company_pos.model',
    'test_corpus_path' : '/home/op/work/survey/data/test.txt',
    'COMPANY_POS'      : 'COMPANY_POS',
    'COMPANY_NEG'      : 'COMPANY_NEG',
    'sentence_path'    : '/home/op/work/survey/data/sentence_sure.txt',
    'embeding_size'    : 100,
    'topn'             : 10,
    'company_name'     : "COMPANY_NAME",

    'clean_config' :
    {
        'keyword_pos': '/home/op/work/survey/data/华夏幸福_keyword_pos.txt',
        'keyword_neg': '/home/op/work/survey/data/华夏幸福_keyword_neg.txt',
        'corpus_in': '/home/op/work/survey/corpus/华夏幸福/华夏幸福_raw.txt',
        'corpus_out' : '/home/op/work/survey/corpus/华夏幸福/华夏幸福_clean.txt'
    },

    'logistic'   :
    {
        'window'       :       5,
        'length'       :      10,
        'class_num'    :       2,
        'in_path'      :      '/home/op/work/survey/data/sentence_sure_extract_10.txt',
        'out_path'     :      '/home/op/work/survey/data/sentence_sure_train_10_10000.txt',
        'company_name' :      'COMPANY_NAME',
        'model_path'   :      '/home/op/work/survey/model/logistic/',
        'test_path'    :      '/home/op/work/survey/data/test_label.txt'
    },

    'lr'        :
    {
        'corpus_path'  :   '/home/op/work/survey/corpus/lr',
        'window'          :   20,
        'topn'            :   5,
    }

}

load_config(conf)

if sys.argv[1] == 'collect':
    print("collecting...")
    fac = CorpusFactory(conf)
    #fac.collect_corpus(sys.argv[2])
    fac.collect_lr_corpus(20)

elif sys.argv[1] == 'w2vec':
    print("train model ...")
    fac = w2vectorFactory(conf)
    fac.model(sys.argv[2])
elif sys.argv[1] == 'test':
    print("test model...")
    disam = Disambiguate(conf)
    disam.test()
elif sys.argv[1] == 'clean':
    print("clean corpus...")
    cleaner = CorpusCleaner(conf)
    cleaner.clean()
elif sys.argv[1] == 'logistic':
    logistic(conf, sys.argv[2])
elif sys.argv[1] == 'lr':
    x_train, y_train, x_test, y_test = get_lr_model_dataset(conf)
    lr = LR_Model()
    lr.train(x_train, y_train)
    lr.test(x_test, y_test)

    svm = SVM_Model()
    svm.train(x_train, y_train)
    svm.test(x_test, y_test)






