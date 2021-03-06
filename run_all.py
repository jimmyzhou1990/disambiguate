import sys
from _corpus.company_corpus import CorpusFactory
from _corpus.clean_corpus import CorpusCleaner
from _corpus.online_test_corpus import OnlineTestCorpus
from _model.word2vector import w2vectorFactory
from _disambiguate.disambiguate import Disambiguate
from _model.lstm_model import Text_LSTM
from _model.blstm_model import BLSTM_WSD
from _model.prepare_data import  get_lstm_dataset
import os

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
    'w2v_model_path'     : '/home/op/work/survey/model/w2vec/',
    'test_corpus_path' : '/home/op/work/survey/data/test.txt',
    'evaluate_corpus'  :  '/home/op/work/survey/corpus/evaluate/evaluate.txt',
    'online_date'      :  {'start':'2018-08-25', 'end':'2018-08-26'},
    'online_path'  :  '/home/op/work/survey/corpus/online/',
    'COMPANY_POS'      : 'COMPANY_POS',
    'COMPANY_NEG'      : 'COMPANY_NEG',
    'sentence_path'    : '/home/op/work/survey/data/sentence_sure.txt',
    'sentence_filter_path'    : '/home/op/work/survey/data/sentence_sure_filter.txt',
    'embeding_size'    : 100,
    'topn'             : 10,
    'company_name'     : "COMPANY_NAME",
    'key_word'         : "COMPANY_NAME",

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
        'test_path'    :   '/home/op/work/survey/data/test_lr.txt',
        'model_path'   :   '/home/op/work/survey/model/lr/',
        'window'          :   50,
        'range'           :   35,
        'topn'            :   5,
    },

    'lstm'     :
    {
        'model_path'    :  '/home/op/work/survey/model/lstm/',
        'corpus_path'   :  '/home/op/work/survey/corpus/lstm/',
        'version'       :  'v3',
        'range'         :  30,
        #'evaluate_corpus'  :  '/home/op/work/survey/corpus/evaluate/evaluate_online_0825.txt',
        #'evaluate_corpus'  :  '/home/op/work/survey/corpus/evaluate/evaluate.txt',
        'evaluate_corpus'  :  '/home/op/work/survey/corpus/evaluate/time',
        'w2vec_version'    :  'v2',
        'domain'           :  'port',
        'method'           :  'mix',
        'batch_size'       :  128,
        'attention'        :  False,
    }

}

load_config(conf)

if sys.argv[1] == 'collect':
    print("collecting...")
    fac = CorpusFactory(conf)
    fac.collect_corpus(sys.argv[2])
    #fac.collect_lstm_corpus()

elif sys.argv[1] == 'collect_online':
    clt = OnlineTestCorpus(conf)
    clt.collect_online_pos()

elif sys.argv[1] == 'w2vec':
    print("train model ...")
    fac = w2vectorFactory(conf)
    fac.model(sys.argv[2])

elif sys.argv[1] == 'test':
    print("test model...")
    if len(sys.argv) != 4:
        print("cmd  test  v3   mix|domain|port")
    conf['lstm']['version'] = sys.argv[2]
    if sys.argv[3] == 'mix' or sys.argv[3] == 'domain':
        conf['lstm']['method'] = sys.argv[3]
    else:
        conf['lstm']['domain'] = sys.argv[3]
        conf['lstm']['method'] = 'single'
    disam = Disambiguate(conf)
    disam.evaluate_models()

elif sys.argv[1] == 'clean':
    print("clean corpus...")
    cleaner = CorpusCleaner(conf)
    cleaner.clean()

elif sys.argv[1] == 'lstm':
    if sys.argv[2] == 'v1':
        conf['lstm']['version'] = 'v1'
        x_train, y_train, x_test, y_test, x_test_info = get_lstm_dataset(conf)
        lstm = Text_LSTM()
        lstm.train_and_test(x_train, y_train, x_test, y_test, x_test_info, 20, 200, conf['lstm']['model_path']+conf['lstm']['version']+'/')
    elif sys.argv[2] == 'v2':
        conf['lstm']['version'] = 'v2'
        x_train, y_train, x_test, y_test, x_test_info = get_lstm_dataset(conf)
        lstm = Text_LSTM()
        lstm.train_and_test(x_train, y_train, x_test, y_test, x_test_info, 22, 1000, conf['lstm']['model_path']+conf['lstm']['version']+'/')
    elif sys.argv[2] == 'v3':
        conf['lstm']['version'] = 'v3'
        x_train, y_train, x_test, y_test, x_test_info = get_lstm_dataset(conf)
        lstm = Text_LSTM()
        lstm.train_and_test(x_train, y_train, x_test, y_test, x_test_info, 22, 100, conf['lstm']['model_path']+conf['lstm']['version']+'/')
    elif sys.argv[2] == 'v4':
        conf['lstm']['version'] = 'v4'
        x_train, y_train, x_test, y_test, x_test_info = get_lstm_dataset(conf)
        lstm = Text_LSTM()
        lstm.train_and_test(x_train, y_train, x_test, y_test, x_test_info,
                            22, 1024, conf['lstm']['model_path']+conf['lstm']['version']+'/')
    elif sys.argv[2] == 'v5':
        conf['lstm']['version'] = 'v5'
        range = conf['lstm']['range']
        x_train, y_train, x_test, y_test, x_test_info, w2vec = get_lstm_dataset(conf)
        lstm = BLSTM_WSD(max_seq_length=range*2, word_keep_prob=1.0, w2vec=w2vec)
        lstm.train_and_test(x_train, y_train, x_test, y_test, x_test_info, 22, 128,
                            conf['lstm']['model_path'] + conf['lstm']['version'] + '/',
                           )

    elif sys.argv[2] == 'v6':  #blstm 分行业训练
        if len(sys.argv) != 4:
            print("argv length: %d" % len(sys.argv))
            print("cmd   lstm    v6     airport|medicine|port|estate|highway")
            exit(-1)
        conf['lstm']['version'] = 'v6'
        conf['lstm']['domain'] = sys.argv[3]
        range = conf['lstm']['range']
        batch_size = conf['lstm']['batch_size']
        path = os.path.join(conf['lstm']['model_path'], conf['lstm']['version'])
        x_train, y_train, x_test, y_test, x_test_info, w2vec = get_lstm_dataset(conf)
        lstm = BLSTM_WSD(max_seq_length=range * 2, batch_size=batch_size, 
                         word_keep_prob=1.0,
                         w2vec=w2vec,
                         model_name=conf['lstm']['domain'],
                         attention=conf['lstm']['attention'],
                         model_path=path)
        lstm.train_and_test(x_train, y_train, x_test, y_test, x_test_info, 15)
                  
