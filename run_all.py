import sys
from _corpus.company_corpus import CorpusFactory
from _corpus.clean_corpus import CorpusCleaner
from _model.train_model import ModelFactory
from _disambiguate.disambiguate import Disambiguate 

def load_config(conf):
    with open("/home/op/work/survey/data/company_sure.txt") as f:
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
    'corpus_path'    : '/home/op/work/survey/corpus/sure/corpus_filter.pos',
    'user_dict'      : '/home/op/work/survey/data/user_dict.txt',
    'stopwords_path' : '/home/op/work/survey/data/stop_word.txt',
    'model_path'     : '/home/op/work/survey/model/company_pos.model',
    'test_corpus_path' : '/home/op/work/survey/data/test.txt',
    'COMPANY_POS'      : 'COMPANY_POS',
    'COMPANY_NEG'      : 'COMPANY_NEG',
    'sentence_path'    : '/home/op/work/survey/data/sentence_sure.txt',

    'clean_config' :
    {
        'keyword_pos': '/home/op/work/survey/data/华夏幸福_keyword_pos.txt',
        'keyword_neg': '/home/op/work/survey/data/华夏幸福_keyword_neg.txt',
        'corpus_in': '/home/op/work/survey/corpus/华夏幸福/华夏幸福_raw.txt',
        'corpus_out' : '/home/op/work/survey/corpus/华夏幸福/华夏幸福_clean.txt'
    },

    'disam_config'   :
    {
        'window'     :      10,
        'topn'       :       5,
    }

}

load_config(conf)

if sys.argv[1] == 'collect':
    print("collecting...")
    fac = CorpusFactory(conf)
    #fac.collect_corpus(sys.argv[2])
    fac.collect_sure_corpus()
elif sys.argv[1] == 'model':
    print("train model ...")
    fac = ModelFactory(conf)
    fac.model(sys.argv[2])
elif sys.argv[1] == 'test':
    print("test model...")
    disam = Disambiguate(conf)
    disam.test()
elif sys.argv[1] == 'clean':
    print("clean corpus...")
    cleaner = CorpusCleaner(conf)
    cleaner.clean()




