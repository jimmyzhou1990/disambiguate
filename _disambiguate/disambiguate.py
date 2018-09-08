import gensim
import jieba
import numpy as np
from _model.lstm_model import Text_LSTM
from _model.blstm_model import BLSTM_WSD
import re
import copy
import pandas as pd
import os

class Disambiguate(object):
    def __init__(self, conf):
        jieba.load_userdict(conf['user_dict'])  # 加载自定义词典
        self.stopword_set = set([l.strip() for l in open(conf['stopwords_path'], 'rt')])
        self.version = conf['lstm']['version']
        self.w2vec_version = conf['lstm']['w2vec_version']
        self.w2v_model = gensim.models.Word2Vec.load(conf['w2v_model_path']+self.w2vec_version+'/company_pos.w2vec')
        self.vocab_set = set(self.w2v_model.wv.vocab)

        self.COMPANY_NEG = conf['COMPANY_NEG']
        self.COMPANY_POS = conf['COMPANY_POS']
        self.COMPANY_NAME = conf['company_name']
        self.method = conf['lstm']['method']
        self.domain = conf['lstm']['domain']

        self.evaluate_corpus = conf['lstm']['evaluate_corpus']
        self.range = conf['lstm']['range']
        self.lstm_model_path = conf['lstm']['model_path']+self.version
        self.company_list =[]
        for company in conf['company_list']:
            self.company_list.append(company['short_name'])
            self.company_list.append(company['full_name'])
        print(self.company_list)

        self.all_domain = ['airport', 'estate', 'food', 'gaoxin', 'highway', 'industry', 'medicine', 'mix', 'port',
                       'tourism', 'yunnanbaiyao']
        self.models = {}

    def load_models(self):
        domain_param = {
                        'airport':{
                            'gate' : 0.5,
                        }, 
                        'estate':{
                            'gate' : 0.5,
                        }, 
                        'food':{
                            'gate' : 0.5,
                        },
                        'gaoxin':{
                            'gate' : 0.5,
                        }, 
                        'highway':{
                            'gate' : 0.5,
                        }, 
                        'industry':{
                            'gate' : 0.5,
                        }, 
                        'medicine':{
                            'gate' : 0.5,
                        }, 
                        'mix':{
                            'gate' : 0.5,
                        }, 
                        'port':{
                            'gate' : 0.5,
                        },
                        'tourism':{
                            'gate' : 0.5,
                        }, 
                        'yunnanbaiyao':{
                            'gate' : 0.5,
                        },
                      }
        m = {
            'model' : None,
            'feed'  : {
                'x_eval' : [],
                'y_eval' : [],
                'x_info' : [],
             },
            'gate' : 0.6,
        }
        for d in self.all_domain:
            mx = copy.deepcopy(m)
            self.models[d] = mx
            self.models[d]['gate'] = domain_param[d]['gate']
            print("domain [%s] gate: %d"%(d, self.models[d]['gate']))
        self.lstm_model = BLSTM_WSD(max_seq_length=self.range*2, word_keep_prob=0.8, w2vec=self.w2v_model)

    def run_models(self):
        p = 0
        rp = 0
        n = 0
        rn = 0
        for d in self.all_domain:
            if self.method == 'single' and d != self.domain:
                continue
            lstm = self.lstm_model
            x_eval = np.array(self.models[d]['feed']['x_eval'])
            y_eval = np.array(self.models[d]['feed']['y_eval'])
            x_info = self.models[d]['feed']['x_info']
            gate = self.models[d]['gate']
            if len(x_eval) == 0:
                continue
            path = self.lstm_model_path+'/'+d
            print('load lstm model: %s'%path)
            pos, rpos, neg, rneg = lstm.evaluate(x_eval, y_eval, x_info, path, d, gate)
            p += pos
            rp += rpos
            n += neg
            rn += rneg
        total = p + n
        print("positive: %d, negtive: %d, recall_pos: %.3f, recall_neg: %.3f, accuracy: %.3f"%(p, n, rp/p, rn/n, (rn+rp)/total))

    def collect_xlsx(self):
        # /home/op/work/survey/log/lstm_eval_badcase_%s.xlsx
        columns = ['company', 'real', 'predict', 'sentence', 'feature word list']
        bad_df = {}
        good_df = {}
        for c in columns:
            bad_df[c] = []
            good_df[c] = []

        for d in self.all_domain:
            path = '/home/op/work/survey/log/lstm_eval_badcase_%s.xlsx' % d
            if os.path.exists(path):
                df = pd.read_excel(path)
                for c in columns:
                    bad_df[c] += list(df[c])
                os.remove(path)
            path = '/home/op/work/survey/log/lstm_eval_goodcase_%s.xlsx' % d
            if os.path.exists(path):
                df = pd.read_excel(path)
                for c in columns:
                    good_df[c] += list(df[c])
                os.remove(path)

        bad_df = pd.DataFrame(bad_df)
        good_df = pd.DataFrame(good_df)
        bad_df.to_excel('/home/op/work/survey/log/lstm_eval_badcase_domain.xlsx', index=False, columns=columns)
        good_df.to_excel('/home/op/work/survey/log/lstm_eval_goodcase_domain.xlsx', index=False, columns=columns)

    def get_domain(self, short_name):
        if self.method == 'mix':
            return 'mix'

        if re.match('.*机场', short_name):
            return 'airport'
        elif re.match('.*[港湾]', short_name):
            return 'port'
        elif re.match('.*高速|.*徐高', short_name):
            return 'highway'
        elif re.match('老百姓', short_name):
            return 'medicine'
        elif re.match('好想你', short_name):
            return 'food'
        elif re.match('.*旅游', short_name):
            return 'tourism'
        elif re.match('.*高新', short_name):
            return 'gaoxin'
        elif re.match('星星|中国盐业|北方工业|电科院', short_name):
            return 'industry'
        elif re.match('云南白药', short_name):
            return 'yunnanbaiyao'
        elif re.match('葛洲坝|华夏幸福|陆家嘴|天地源|阳光城|金融街|华侨城|花样年|新华联|浦东金桥', short_name):
            return 'estate'
        else:
            return 'mix'

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

    def filter_word(self, word, vocab, short_name):
        if word == short_name or word == ' ' or word in self.stopword_set:
            return ''

        #pattern_str = '^[年月日]$'
        #pattern = re.compile(pattern_str)
        #res = pattern.match(word)
        #if res:
        #    return 'UnknownWord'

        pattern_str = '^\d{6}$'  # 股票代码
        pattern = re.compile(pattern_str)
        res = pattern.match(word)
        if res:
            pass

        # pattern_str = '''[0-9]|[a-z]|[A-Z]|^[年月日中]$|】|【|前|后|上午|
        # 再|原|一个|不断|时间|时|记者|获悉|.*网|报道|―|全国|相关|新|正式|全|本报讯|
        # 一天|以来|称|刚刚|查看|
        # 已|今天|近期|有望|一直|继续|昨天|预计'''
        else:
            pattern_str = '\d+\.*\d*%*'
            pattern = re.compile(pattern_str)
            res = pattern.match(word)
            if res:
                return '8'

        if word not in vocab:
            return 'UnknownWord'

        return word
    
    # 过滤句子
    def filter_sentence(self, s, keyword):
        s = s.replace(keyword, ' '+keyword+' ')
        splits = re.split('[@|]', s)
        for ss in splits:
            if ss.find(keyword) >= 0:
                return ss
        return ""

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
                    print(items)
                    print("eval corpus bad line: short-name    text    label")
                    continue

                short_name, text, label = items
                text = self.filter_sentence(text, short_name)
                wordlist = [w for w in list(jieba.cut(text))]
                #print("wordlist: %s"%wordlist)
                try:
                    keyword_position = wordlist.index(short_name)
                except:
                    print(text)
                    print("cannot find key word[%s] in word list: [%s]"%(short_name, wordlist))
                    continue

                domain = self.get_domain(short_name)
                # preceding
                pre_veclist = []
                pre_wordlist = []
                for w in wordlist[keyword_position::-1]:
                    w = self.filter_word(w, vocab_set, short_name)
                    if w:
                        pre_wordlist.insert(0, w)
                        pre_veclist.insert(0, w2vec[w])
                    if len(pre_wordlist) >= range:
                        break
                if len(pre_wordlist) < range:  # padding 0
                    padding = [np.zeros(100)] * (range - len(pre_wordlist))
                    pre_veclist += padding

                # succeeding
                suc_veclist = []
                suc_wordlist = []
                for w in wordlist[keyword_position:]:
                    w = self.filter_word(w, vocab_set, short_name)
                    if w:
                        suc_wordlist.insert(0, w)
                        suc_veclist.insert(0, w2vec[w])
                    if len(suc_wordlist) >= range:
                        break
                if len(suc_wordlist) < range:  # padding 0
                    padding = [np.zeros(100)] * (range - len(suc_wordlist))
                    suc_veclist += padding

                # concat
                veclist = pre_veclist + suc_veclist
                info = (short_name, "".join(wordlist), pre_wordlist , suc_wordlist)

                x_eval.append(veclist)
                y_eval.append([1, 0] if label == '1' else [0, 1])
                x_info.append(info)

                self.models[domain]['feed']['x_eval'].append(veclist)
                self.models[domain]['feed']['y_eval'].append([1, 0] if label == '1' else [0, 1])
                self.models[domain]['feed']['x_info'].append(info)

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
        self.load_models()

        x_eval, y_eval, x_info = self.load_sentence_feature(self.range, 2*self.range,
                                                             self.w2v_model, self.vocab_set)

        self.run_models()
        self.collect_xlsx()

    def evaluate_models(self):
        self.evaluate_lstm_model()
