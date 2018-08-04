import jieba
import sys
import gensim
from gensim.models.word2vec import Word2Vec
import re

class ModelFactory(object):
    def __init__(self, conf):
        jieba.load_userdict(conf['user_dict'])  # 加载自定义词典
        self.corpus_path = conf['corpus_path']
        self.stopwords_path = conf['stopwords_path']
        self.model_path = conf['model_path']
        self.sentence_path = conf["sentence_path"]
        self.corpus = []
        self.sentences = []

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
        with open(self.sentence_path, "r") as f:
            for line in f:
                self.sentences.append([w for w in line.strip().split(" ")])


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
            model = gensim.models.Word2Vec.load(self.model_path)
            print(model.most_similar('COMPANY_NAME', topn=100))

        elif cmd == 'all':
            self.load_sure_corpus()
            self.cut()
            self.load_sentence()
            self.word2vec()
            model = gensim.models.Word2Vec.load(self.model_path)
            print(model.most_similar('COMPANY_NAME', topn=100))

        else:
            print("invalid cmd: train|cut|test")
