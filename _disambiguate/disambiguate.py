import gensim
import jieba
import numpy as np
from _model.lstm_model import Text_LSTM
from _model.blstm_model import BLSTM_WSD
import re
import copy
import pandas as pd
import os
import time 
import datetime

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
        self.attention = conf['lstm']['attention']

        self.evaluate_corpus = conf['lstm']['evaluate_corpus']
        self.range = conf['lstm']['range']
        self.lstm_model_path = os.path.join(conf['lstm']['model_path'], self.version)
        self.company_list =[]
        for company in conf['company_list']:
            self.company_list.append(company['short_name'])
            self.company_list.append(company['full_name'])
        print(self.company_list)

        self.all_domain = ['airport', 'estate', 'food', 'gaoxin', 'highway', 'industry', 'medicine', 'mix', 'port',
                       'tourism', 'yunnanbaiyao', 'resource', 'concrete']
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
                        'resource':{
                            'gate' : 0.5,
                        },
                        'concrete':{
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
            self.models[d]['model']= BLSTM_WSD(max_seq_length=self.range*2,
                                             word_keep_prob=0.8, 
                                             w2vec=self.w2v_model, 
                                             model_name=d,
                                             attention=self.attention,
                                             model_path=self.lstm_model_path)

    def run_models(self):
        p = 0
        rp = 0
        n = 0
        rn = 0
        for d in self.all_domain:
            if self.method == 'single' and d != self.domain:
                continue
            lstm = self.models[d]['model']
            
            x_eval = np.array(self.models[d]['feed']['x_eval'])
            y_eval = np.array(self.models[d]['feed']['y_eval'])
            x_info = self.models[d]['feed']['x_info']
            gate = self.models[d]['gate']
            if len(x_eval) == 0:
                continue
            #print('load lstm model: %s'%path)
            pos, rpos, neg, rneg = lstm.evaluate(x_eval, y_eval, x_info, d, gate)
            p += pos    #正例
            rp += rpos  #正例正确召回
            n += neg    #负例
            rn += rneg  #负例正确召回
        total = p + n   #总case数
        rpn = rp + n - rn # 判定为正例的总数
        #print("positive: %d, negtive: %d, recall_pos: %.3f, precision_pos: %.3f, total_accuracy: %.3f"%(p, n, rp/(p+0.01), rp/(rpn+0.01), (rn+rp)/total))

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
        elif re.match('鄂尔多斯', short_name):
            return 'resource'
        elif re.match('祁连山|万年青', short_name):
            return 'concrete'
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

    # [(short_name, title:xxxx), ()]
    def load_from_list(self, batch_input, range,  w2vec, vocab_set):
        x_eval = []
        y_eval = []
        x_info = []
        for domain in self.all_domain:
            self.models[domain]['feed']['x_eval'] = []
            self.models[domain]['feed']['y_eval'] = []
            self.models[domain]['feed']['x_info'] = []
        for short_name, title, label in batch_input:
            title = self.filter_sentence(title, short_name)
            wordlist = [w for w in list(jieba.cut(title))]
            try:
                keyword_position = wordlist.index(short_name)
            except:
                print(title)
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

        #print('evaluate corpus length: %d'%len(x_info))
        # print('domian: %s, len %d'%(domain, len(self.models[domain]['feed']['x_eval'])))
        return np.array(x_eval), np.array(y_eval), x_info

    def generate_input_list(self, i):
        path = os.path.join(self.evaluate_corpus, 'evaluate_online_0825_%d.txt'%i)
        test_list = []
        with open(path, 'r') as f:
            for l in f:
                items = l.strip().split('\t') 
                if len(items) != 3:
                    print(items)
                    print("eval corpus bad line: short-name    text    label")
                    continue
                short_name, text, label = items
                s = (short_name, text, label)
                test_list.append(s)
        return test_list

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

        #print('evaluate corpus length: %d'%len(x_info))
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

    def evaluate_model_time(self):
        s_t = 1000*time.time() 
        self.load_models()
        e_t = 1000*time.time() 
        print("load models cost: %d ms"%(e_t-s_t))

        summary_time = []
        for i in range(1001)[1:-1:1]:
            batch_list = self.generate_input_list(i)
            s_t = 1000*time.time()
            self.load_from_list(batch_list, self.range, self.w2v_model, self.vocab_set) 
            self.run_models()
            e_t = 1000*time.time() 
            time_cost = e_t - s_t
            cost_aver = time_cost/i
            summary_time.append((str(i), str(time_cost), str(cost_aver)))
            print("batch_num: %d, cost time: %dms, average cost: %.3f"%(i, time_cost, cost_aver))
        with open(os.path.join(self.evaluate_corpus, 'summary_time.txt'), 'w+') as f:
            for s in summary_time:
                l = '\t'.join(s)+'\n'
                f.write(l)

    def evaluate_batch(self):
        s_t = 1000*time.time() 
        self.load_models()
        e_t = 1000*time.time() 
        print("load models cost: %d ms"%(e_t-s_t))
        batch_num = [10, 40, 50, 100, 200, 400, 500, 1000, 2000, 3000, 4000,  5000, 6000, 7000, 8000, 9000,  10000]
        path = os.path.join(self.evaluate_corpus, 'evaluate_online_0825_1w.txt')
        with open(path, "r") as f:
            test_list = []
            for l in f:
                items = l.strip().split('\t') 
                if len(items) != 3:
                    print(items)
                    print("eval corpus bad line: short-name    text    label")
                    continue
                short_name, text, label = items
                s = (short_name, text, label)
                test_list.append(s)
        print("test_list length: %s"%len(test_list))
        batch_lists = [[test_list[i:i+b] for i in range(0, len(test_list), b)] for b in batch_num] 

        times = []
        for i in range(len(batch_num)):
            li = batch_lists[i]
            time_cost = 0
            for lli in li:
                s_t = 1000*time.time()
                self.load_from_list(lli, self.range, self.w2v_model, self.vocab_set)
                self.run_models()
                e_t = 1000*time.time() 
                time_cost += (e_t - s_t)
            print("batch_num: %s, cost time: %.3f"%(batch_num[i], time_cost))
            times.append((str(batch_num[i]), str(time_cost)))

        with open(os.path.join(self.evaluate_corpus, 'summary_time_batch.txt'), 'w+') as f:
            for s in times:
                l = '\t'.join(s)+'\n'
                f.write(l)

    def evaluate_lstm_model(self):

        self.load_models()

        x_eval, y_eval, x_info = self.load_sentence_feature(self.range, 2*self.range,
                                                             self.w2v_model, self.vocab_set)
        
        s_t = 1000*time.time()
        self.run_models()
        e_t = 1000*time.time() 
        print("cost time: %dms"%(e_t-s_t))

        self.collect_xlsx()

    def evaluate_models(self):
        #self.evaluate_lstm_model()
        #self.evaluate_model_time()
        self.evaluate_batch()
        
