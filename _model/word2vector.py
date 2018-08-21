import jieba
import sys
import gensim
from gensim.models.word2vec import Word2Vec
import re

class w2vectorFactory(object):
    def __init__(self, conf):
        jieba.load_userdict(conf['user_dict'])  # 加载自定义词典
        self.corpus_path = conf['corpus_path']
        self.stopwords_path = conf['stopwords_path']
        self.model_path = conf['w2v_model_path']+'v1'+'/company_pos.w2vec'
        self.sentence_path = conf["sentence_path"]
        self.sentence_filter_path = conf['sentence_filter_path']
        self.keyword = conf['key_word']
        self.corpus = []
        self.sentences = []
        self.company_list =[]
        for company in conf['company_list']:
            self.company_list.append(company['short_name'])
            self.company_list.append(company['full_name'])

    #加载pos和neg, 取title和content, 合并成一个list
    def load_corpus(self):
        with open(self.corpus_path+'/company.pos', 'rt') as f:
            self.corpus += [' '.join(l.strip().split('\t')[1:-1]) for l in f]
        with open(self.corpus_path+'/company.neg', 'rt') as f:
            self.corpus += [' '.join(l.strip().split('\t')[1:-1]) for l in f]

    def load_sure_corpus(self):
        corpus = self.corpus_path
        print("load corpus: %s"%corpus)
        with open(corpus, 'r') as f:
            pattern = re.compile('\n|\t')
            self.corpus = [self.clean_line(pattern, ' ', l) for l in f]
        print("load corpus finished. size=%d"%len(self.corpus))

    def load_sentence(self):
        def filter_sentence(sentence):
            pass
        f_filter = open(self.sentence_filter_path, 'w+')
        with open(self.sentence_path, "r") as f:
            for line in f:
                #self.sentences.append([w for w in line.strip().split(" ")])
                sentence = line.strip()
                sentence = re.sub('\d+\.\d+%*', '8', sentence)
                sentence = re.sub("\D\d{1,4}\D|\d{7,}", ' 8 ', sentence)  #替换所有数字(整数/小数/百分数)为8
                sentence = re.sub(' {2,}', ' ', sentence)
                f_filter.write(sentence+'\n')
                self.sentences.append([w for w in line.strip().split(" ")])
        f_filter.close()

    def clean_line(self, pattern, s, l):
        l = pattern.sub(s, l)
        return l

    #分词
    def cut(self):
        print("start to cut sentence...")
        stopword_set = set([l.strip() for l in open(self.stopwords_path, 'rt')])

        sentence_file = open(self.sentence_path, 'w+')
        count = 0
        total = len(self.corpus)
        for l in self.corpus:
            try:
                wordList = list(jieba.cut(l))
                self.sentences.append([w for w in wordList if w not in stopword_set])
                sentence_file.write(" ".join(self.sentences[-1]) + "\n")
                count += 1
                if count % 100 == 0:
                    print("cut %d/%d line"%(count, total))
            except Exception as e:
                print("jieba cut error!")
                print(l)

        sentence_file.close()
        print("cut finished.")

    #训练模型
    def word2vec(self):
        print("start to train model...")
        model = gensim.models.Word2Vec(self.sentences, size=100, min_count=10, window=5, workers=10)
        model.save(self.model_path)
        print("train model finished.")

    def test_model(self):
        model = gensim.models.Word2Vec.load(self.model_path)
        print(model.most_similar('COMPANY_NAME', topn=100))

    def model(self, cmd):
        if cmd == 'train':
            self.load_sentence()
            self.word2vec()

        elif cmd == 'cut':
            self.load_sure_corpus()
            self.cut()

        elif cmd == 'test':
            worlist = ['电脑', '银行', '上海', '汽车', '人民币',
                       '新闻', '大厦', '股票', '上市', '公路',
                       '动物园', '地铁', '蓝天', '春节', '公司', '习近平', '长青', '云南白药', '白云机场']
            model = gensim.models.Word2Vec.load(self.model_path)
            vocab_set = set(model.wv.vocab)
            #print(model.most_similar(self.keyword, topn=500))

            for w in self.company_list:
                if w in vocab_set:
                    print("topn similarity with [%s]"%w)
                    print(model.most_similar(w, topn=50))

        elif cmd == 'all':
            self.load_sure_corpus()
            self.cut()
            self.load_sentence()
            self.word2vec()
            model = gensim.models.Word2Vec.load(self.model_path)
            print(model.most_similar(self.keyword, topn=1000))

        else:
            print("invalid cmd: train|cut|test")
