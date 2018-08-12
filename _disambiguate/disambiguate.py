import gensim
import jieba
import numpy as np
from _model.lstm_model import Text_LSTM

class Disambiguate(object):
    def __init__(self, conf):
        jieba.load_userdict(conf['user_dict'])  # 加载自定义词典
        self.w2v_model = gensim.models.Word2Vec.load(conf['w2v_model_path'])
        self.vocab_set = set(self.w2v_model.wv.vocab)
        self.test_corpus_path = conf['test_corpus_path']
        self.COMPANY_NEG = conf['COMPANY_NEG']
        self.COMPANY_POS = conf['COMPANY_POS']
        self.COMPANY_NAME = conf['company_name']
        self.cases = []
        self.topn = conf['topn']
        self.evaluate_corpus = conf['evaluate_corpus']
        self.lr_range = conf['lr']['range']
        self.stopword_set = set([l.strip() for l in open(conf['stopwords_path'], 'rt')])
        self.lr_modelpath = conf['lr']['model_path']
        self.lstm_modelpath = conf['lstm']['model_path']

    def find(self, s, m):
        i = 0
        posi = []
        while i < len(s):
            p = s.find(m, i)
            if p >= 0:
                posi.append(p)
                i = p + len(m)
            else:
                break

        return posi



    def load_test_corpus(self):
        with open(self.test_corpus_path, "rt") as f:
            for line in f:
                items = line.strip().split('\t')
                case = {}
                case['short_name'] = items[0]
                case['sentence'] = items[1]
                case['word_list'] = list(jieba.cut(items[1]))
                #print(case)
                self.cases.append(case)

    def cal_each_similarity(self, case):
        similarity_list = []
        for index, w in enumerate(case['word_list']):
            if w not in self.vocab_set or w == case['short_name']:
                continue
            else:
                similarity_list.append((index, w, self.w2v_model.similarity(w, self.COMPANY_NAME)))
        return similarity_list
  
    def cal_similarity(self, case):
        sum_p = .0
        sum_n = .0
        count = 0.01
        for w in case['word_list']:
            if w not in self.vocab_set or w == case['short_name']:
                continue
            count += 1
            sim_p = self.w2v_model.similarity(w, self.COMPANY_POS)
            sim_n = self.w2v_model.similarity(w, self.COMPANY_NEG)
            print("%s(p:%f, n:%f)"%(w, sim_p, sim_n), end="   ")
            sum_p += sim_p
            sum_n += sim_n
        print("")
        return sum_p/count, sum_n/count

    def sort_similarity(self, similarity_list):
        res = sorted(similarity_list, key=lambda x: x[2], reverse=True)
        return res

    def find_shortname_position(self, case):
        for index, w in enumerate(case['word_list']):
            if w == case['short_name']:
                return index
        return -1

    def cal_similarity_with_offset(self, similarity_list, position):
        max_diff = 0
        for index, _, simi in similarity_list:
            diff = position-index if position >= index else index-position
            max_diff = diff if diff>max_diff else max_diff

        similarity_list_offset = []
        for index, w, simi in similarity_list:
            diff = position - index if position >= index else index - position
            weight = (max_diff-diff+1)/max_diff
            similarity_list_offset.append((index, w, simi, weight, weight * simi))

        return  similarity_list_offset

    def summary_similarity(self, similarity_list_final):
        sum = .0
        for index, _, _, _, simi in similarity_list_final:
            sum += simi
        return sum / len(similarity_list_final)


    def test(self):
        self.load_test_corpus()
        for case in self.cases:
            print(case['word_list'])

            position = self.find_shortname_position(case)
            if position >= 0:
                print('%s found at %d'%(case['short_name'], position))
            else:
                print('cannot find keyword: %s'%case['short_name'])
                position = 0

            similarity_list = self.cal_each_similarity(case)  # [(index, simi), ...]
            similarity_list_sorted = self.sort_similarity(similarity_list)
            #print(similarity_list_sorted)

            similarity_list_topn = similarity_list_sorted[:self.topn]
            #print('topn similarity:')
            #print(similarity_list_topn)
            similarity_list_final = self.cal_similarity_with_offset(similarity_list_topn, position)

            print('topn similarity with weight:')
            print(similarity_list_final)

            similarity_summary = self.summary_similarity(similarity_list_final)

            print("[%s]---%s"%(case['short_name'], case['sentence']))
            print("similarity with COMPANY_POS: %f"%similarity_summary)
            print("---------------------------------------------------------------------------------")

    def filter_word(self, word):
        import re
        if word == 'COMPANY_NAME' or word == 'COMPANY_POS' or word == 'COMPANY_NEG':
            return False

        pattern_str = '''[0-9]|[a-z]|[A-Z]|月|年|日|中|】|【|前|后|上午|
        再|原|一个|不断|时间|时|记者|获悉|.*网|报道|―|全国|相关|新|正式|全|本报讯|
        一|以来|称|上海|深圳|广州|重庆|北京|苏州|南京|杭州|武汉|江苏|国际|刚刚|查看|
        已|今天|近期|有望|一直|继续|昨天|五|预计|丨''';
        # print("feature filter patterh: %s"%pattern_str)

        pattern = re.compile(pattern_str)
        res = pattern.search(word)

        if res:
            return False
        return True

    def get_feature_average(self, wordlist, range, short_name, position, w2vec, vocab_set):
        vecsum = np.zeros(100)
        feature_word = []

        for index, w in enumerate(wordlist):
            if w == short_name or not self.filter_word(w) or w not in vocab_set:
                continue

            if index < position - range or index > position + range:
                continue

            vecsum += w2vec[w]
            feature_word.append(w)

        return vecsum, (short_name, "".join(wordlist), feature_word)


    def lr_load_eval_corpus(self):
        x_eval = []
        x_info = []
        y_eval = []
        with open(self.evaluate_corpus, 'r') as f:
            for l in f:
                items = l.strip().split('\t')
                if len(items) != 3:
                    print("eval corpus bad line: short-name    text    label")
                    continue

                short_name, text, label = items
                wordlist = [w for w in list(jieba.cut(text))]

                try:
                    position = wordlist.index(short_name)
                except:
                    print("cannot find key word[%s] in word list: [%s]"%(short_name, wordlist))
                    continue

                vecsum, text_info = self.get_feature_average(wordlist, self.lr_range, short_name,
                                         position, self.w2v_model, self.vocab_set)

                x_eval.append(vecsum)
                x_info.append(text_info)
                y_eval.append(1 if label == '1' else 0)

        return np.array(x_eval), np.array(y_eval), x_info


    def evaluate_lr_model(self):
        import pickle

        x_eval, y_eval, x_info = self.lr_load_eval_corpus()

        with open(self.lr_modelpath+'/lr.model', 'rb') as f:
            lr = pickle.load(f)
        print(type(lr))

        lr.test(x_eval, y_eval, x_info)

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

                try:
                    keyword_position = wordlist.index(short_name)
                except:
                    print("cannot find key word[%s] in word list: [%s]"%(short_name, wordlist))
                    continue

                veclist = []
                feature_wlist = []
                for index, w in enumerate(wordlist):
                    if index < keyword_position - range or index > keyword_position + range:
                        continue
                    if w in vocab_set and self.filter_word(w):
                        veclist.append(w2vec[w])
                        feature_wlist.append(w)
                if len(veclist) < seq_length:  # padding 0
                    padding = [np.zeros(100)] * (seq_length - len(veclist))
                    veclist = veclist + padding

                x_eval.append(veclist)
                y_eval.append([1, 0] if label == '1' else [0, 1])
                x_info.append((short_name, "".join(wordlist), feature_wlist))

        return np.array(x_eval), np.array(y_eval), x_info

    def evaluate_lstm_model(self):
        x_eval, y_eval, x_info = self.load_sentence_feature(self.lr_range, 2*self.lr_range,
                                                            self.w2v_model, self.vocab_set)
        lstm = Text_LSTM()
        lstm.evaluate(x_eval, y_eval, x_info, self.lstm_modelpath)

    def evaluate_models(self, model):
        if model == 'lr':
            self.evaluate_lr_model()
        elif model == 'lstm':
            self.evaluate_lstm_model()
