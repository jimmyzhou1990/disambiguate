import gensim
import jieba
import numpy as np
from _model.lstm_model import Text_LSTM
import re

class Disambiguate(object):
    def __init__(self, conf):
        jieba.load_userdict(conf['user_dict'])  # 加载自定义词典
        self.stopword_set = set([l.strip() for l in open(conf['stopwords_path'], 'rt')])
        self.version = conf['lstm']['version']
        self.w2v_model = gensim.models.Word2Vec.load(conf['w2v_model_path']+'v1'+'/company_pos.w2vec')
        self.vocab_set = set(self.w2v_model.wv.vocab)

        self.COMPANY_NEG = conf['COMPANY_NEG']
        self.COMPANY_POS = conf['COMPANY_POS']
        self.COMPANY_NAME = conf['company_name']

        self.evaluate_corpus = conf['lstm']['evaluate_corpus']
        self.range = conf['lstm']['range']
        self.lstm_model_path = conf['lstm']['model_path']+self.version
        self.company_list =[]
        for company in conf['company_list']:
            self.company_list.append(company['short_name'])
            self.company_list.append(company['full_name'])
        print(self.company_list)

    def is_strange_charactor(self, word):
        strange_charactor = ['\u3000', '\u200b', '\u2002', '\u2003', '\u200c', '\u202a', '\u202c',
                             '\ufeff', '\ue8c3', '\uf8f5', '\ue587', '\ue31c', '\ue193', '\ue033',
                             '\ue14b', '\ue1a9', '\ue604',
                             '\xa0', '\x08', '\x07', '\x00', '\xad', '\x0b', ]
        if word in strange_charactor:
            return True

        ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
        if ILLEGAL_CHARACTERS_RE.findall(word):
            return True

        return False

    def is_company_name(self, word):
        return word in self.company_list

    def filter_word(self, word):
        import re
        if word == 'COMPANY_NAME' or word == 'COMPANY_POS' or word == 'COMPANY_NEG':
            return False

        pattern_str = '^\d{6}$'   #股票代码
        pattern = re.compile(pattern_str)
        res = pattern.match(word)
        if res:
            return True

        pattern_str = '''[0-9]|[a-z]|[A-Z]|^[年月日中]$|】|【|前|后|上午|
        再|原|一个|不断|时间|时|记者|获悉|.*网|报道|―|全国|相关|新|正式|全|本报讯|
        一天|以来|称|刚刚|查看|
        已|今天|近期|有望|一直|继续|昨天|预计'''

        pattern = re.compile(pattern_str)
        res = pattern.search(word)

        if res:
            return False
        return True

    def extend_vector(self, shortname, vec):
        if shortname == '老百姓':
            v1 = np.array([0, 0, 0, 1.])
            vec = np.concatenate((vec, v1))
        elif shortname == '白云机场':
            v2 = np.array([0, 0, 1., 0])
            vec = np.concatenate((vec, v2))
        elif shortname == '华夏幸福':
            v3 = np.array([0, 1., 0, 0])
            vec = np.concatenate((vec, v3))
        elif shortname == '好想你':
            v3 = np.array([1., 0, 0, 0])
            vec = np.concatenate((vec, v3))
        else:
            v0 = np.array([0, 0, 0, 0])
            vec = np.concatenate((vec, v0))
        return vec

    def load_sentence_feature(self, range, seq_length, w2vec, vocab_set):
        x_eval = []
        y_eval = []
        x_info = []
        with open(self.evaluate_corpus, 'r') as f:
            for l in f:
                items = l.strip().split('\t')
                if len(items) != 3:
                    print("eval corpus bad line: short-name    text    label")
                    continue

                short_name, text, label = items
                wordlist = [w for w in list(jieba.cut(text))]
                print("wordlist: %s"%wordlist)
                try:
                    keyword_position = wordlist.index(short_name)
                except:
                    print("cannot find key word[%s] in word list: [%s]"%(short_name, wordlist))
                    continue

                veclist = []
                feature_wlist = []
                for index, w in enumerate(wordlist):
                    if index < keyword_position - range or index > keyword_position + range or w == short_name:
                        continue
                    if w in vocab_set and self.filter_word(w) and not self.is_strange_charactor(w):
                        #w_extend = self.extend_vector(short_name, w2vec[w])
                        veclist.append(w2vec[w])
                        #veclist.append(w_extend)
                        feature_wlist.append(w)
                if len(veclist) < seq_length:  # padding 0
                    padding = [np.zeros(100)] * (seq_length - len(veclist))
                    veclist = veclist + padding

                if len(feature_wlist) >= 2:
                    x_eval.append(veclist)
                    y_eval.append([1, 0] if label == '1' else [0, 1])
                    x_info.append((short_name, "".join(wordlist), feature_wlist))
        print('evaluate corpus length: %d'%len(x_info))
        return np.array(x_eval), np.array(y_eval), x_info

    def load_filter7_log(self, range, seq_length, w2vec, vocab_set):
        x_eval = []
        y_eval = []
        x_info = []
        with open(self.evaluate_corpus, 'r') as f:
            for l in f:
                items = l.strip().split('\001')
                if len(items) != 3:
                    print("eval corpus bad line: title    label")
                    continue

                short_name, title, label = items
                #print(items)
                wordlist = [w for w in list(jieba.cut(title))]

                veclist = []
                feature_wlist = []
                for index, w in enumerate(wordlist):
                    if w in vocab_set and self.filter_word(w) and w != short_name and not self.is_strange_charactor(w):
                        veclist.append(w2vec[w])
                        feature_wlist.append(w)
                if len(veclist) < seq_length:  # padding 0
                    padding = [np.zeros(100)] * (seq_length - len(veclist))
                    veclist = veclist + padding
                else:
                    veclist = veclist[:seq_length] #截断

                if len(feature_wlist) >= 2:
                    x_eval.append(veclist)
                    y_eval.append(np.array([1, 0]) if label == '1' else np.array([0, 1]))
                    x_info.append((short_name, "".join(wordlist), feature_wlist))

        return np.array(x_eval), np.array(y_eval), x_info

    def evaluate_lstm_model(self):
        #x_eval, y_eval, x_info = self.load_filter7_log(self.range, 2*self.range,
        #                                                    self.w2v_model, self.vocab_set)

        x_eval, y_eval, x_info = self.load_sentence_feature(self.range, 2*self.range,
                                                             self.w2v_model, self.vocab_set)

        #print(x_eval[0])
        #print(type(x_eval))
        #print(y_eval[0])
        #print(type(y_eval[0]))

        lstm = Text_LSTM()
        lstm.evaluate(x_eval, y_eval, x_info, self.lstm_model_path)

    def evaluate_models(self):
        self.evaluate_lstm_model()
