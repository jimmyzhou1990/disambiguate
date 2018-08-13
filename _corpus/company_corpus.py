import sys
from poa.pipline import *
from poa.es import *
import copy
import jieba
import re

class CorpusFactory(object):
    def __init__(self, conf):
        self.company_list = conf['company_list']
        self.corpus_path = conf['corpus_path']
        self.COMPANY_NEG = conf['COMPANY_NEG']
        self.COMPANY_POS = conf['COMPANY_POS']
        self.stopword_path = conf['stopwords_path']
        self.window = conf['lr']['window']
        jieba.load_userdict(conf['user_dict'])  # 加载自定义词典


    def delete_some_words(self, delwords, srcwords):
        newwords = [w for w in srcwords if w not in delwords]
        return newwords

    def collect_lr_corpus(self):
        #f_pos = open("%s/lr/lr.pos"%self.corpus_path, 'w+')
        #f_neg = open("%s/lr/lr.neg"%self.corpus_path, 'w+')
        #fcut_pos = open("%s/lr/lr_cut.pos"%self.corpus_path, 'w+')
        #fcut_neg = open("%s/lr/lr_cut.neg"%self.corpus_path, 'w+')
        #fextract_pos = open("%s/lr/extract_%d_lr_cut.pos"%(self.corpus_path, self.window), 'w+')
        #fextract_neg = open("%s/lr/extract_%d_lr_cut.neg" % (self.corpus_path, self.window), 'w+')
        f_title_pos = open("%s/lr/lr_title.pos"%self.corpus_path, 'w+')
        f_title_neg = open("%s/lr/lr_title.neg" % self.corpus_path, 'w+')
        min_len = 20
        each = 100

        key_word = ['股票', '股吧', '上市', '股份', '经营', '大盘', '成交金额', '投资', '跌停',
                    '负债', '发债', '债券', '控股', '公司', '融资', '涨停', '利润', '集团',
                    '证券', '下跌', '大跌', '收盘', '深股', '港股', '跌了', '跌出', '亏损',
                    '复牌', '涨跌', '停牌', '复盘', '只股', '成交', '公告', '科技股', '每股',
                    '股东', '品牌', '枣', '交易所', '买入', '卖出', '柿', '地产', '缩股', '市值',
                    '加仓', '000022', '000620', '股市', ]

        strange_charactor = ['\u3000', '\u200b', '\u2002', '\u2003', '\u200c', '\u202a', '\u202c',
                             '\ufeff', '\ue8c3', '\uf8f5', '\ue587', '\ue31c', '\ue193', '\ue033',
                             '\ue14b', '\ue1a9', '\ue604',
                             '\xa0', '\x08', '\x07', '\x00', '\xad', '\x0b', ]

        stopword_set = set([l.strip() for l in open(self.stopword_path, 'rt')])

        #收集负例语料
        for company in self.company_list:
            short_name = company['short_name']
            #collect neg
            print("[%s]collect lr_neg corpus..."%short_name)
            with open("%s/%s/%s_neg.txt" % (self.corpus_path, short_name, short_name), 'r') as f:
                count = 0
                for l in f:
                    items = l.strip().split("\t")
                    title = items[1]
                    text = items[2]

                    conflag = False
                    for k in key_word:
                        if text.find(k) >= 0 or title.find(k) >= 0:
                            conflag = True

                    if len(title) < min_len or text.find(self.COMPANY_NEG) < 0 or conflag:
                        continue

                    #f_neg.write('\t'.join(items)+"\n")
                    #wordlist = [w for w in list(jieba.cut(title+' '+text)) if w not in stopword_set]
                    #wordlist = [w for w in list(jieba.cut(title + ' ' + text))]
                    wordlist = [w for w in list(jieba.cut(title))]
                    wordlist = self.delete_some_words(strange_charactor, wordlist)
                    #print(wordlist)
                    wordlist_sentence = " ".join(wordlist)
                    wordlist_sentence = re.sub(r' +', ' ', wordlist_sentence)  #替换多个空格为1个
                    #fcut_neg.write(wordlist_sentence+"\n")

                    #提取
                    # extract = re.search('([^ ]+ ){0,%d}%s( [^ ]+){%d}'%(self.window, self.COMPANY_NEG, self.window), wordlist_sentence, flags=0)
                    # if extract:
                    #     fextract_neg.write(short_name+"\t"+extract.group()+"\n")
                    f_title_pos.write(short_name+'\t'+wordlist_sentence+'\n')
                    count += 1
                    if count > each:  #每个公司收集50句
                        break
                print("collect %s lines"%count)
        #f_neg.close()
        #fcut_neg.close()
        #fextract_neg.close()
        f_title_pos.close()

        #收集正例语料
        for company in self.company_list:
            short_name = company['short_name']
            #collect pos
            print("[%s]collect lr_pos corpus..."%short_name)
            with open("%s/%s/%s_pos.txt" % (self.corpus_path, short_name, short_name), 'r') as f:
                count = 0
                for l in f:
                    items = l.strip().split("\t")
                    title = items[1]
                    text = items[2]

                    if len(title) < min_len or text.find(self.COMPANY_POS) < 0:
                        continue

                    #f_pos.write("\t".join(items)+'\n')
                    #wordlist = [w for w in list(jieba.cut(title+' '+text)) if w not in stopword_set]
                    #wordlist = [w for w in list(jieba.cut(title + ' ' + text))]
                    wordlist = [w for w in list(jieba.cut(title))]
                    wordlist = self.delete_some_words(strange_charactor, wordlist)
                    #print(wordlist)
                    wordlist_sentence = " ".join(wordlist)
                    wordlist_sentence = re.sub(r' +', ' ', wordlist_sentence)  # 替换多个空格为1个 '\u2002'
                    #fcut_pos.write(wordlist_sentence + "\n")
                    f_title_neg.write(short_name+'\t'+wordlist_sentence+'\n')
                    #提取
                    # extract = re.search('([^ ]+ ){0,%d}%s( [^ ]+){%d}'%(self.window, self.COMPANY_POS, self.window), wordlist_sentence, flags=0)
                    # if extract:
                    #     fextract_pos.write(short_name+"\t"+extract.group()+"\n")

                    count += 1
                    if count > each:  #每个公司收集50句
                        break
                print("collect %s lines"%count)
        #f_pos.close()
        #fcut_pos.close()
        #fextract_pos.close()
        f_title_neg.close()

    def collect_sure_corpus(self):
        for company in self.company_list:
            full_name = company['full_name']
            short_name = company['short_name']
            docs = self.read_es(full_name)
            with open("%s/%s_raw.txt"%(self.corpus_path, short_name), 'w+') as f:
                for doc in docs:
                    f.write("%s\t%s\t%s\t%s\n"%(full_name, short_name, doc['_source']['title'], doc['_source']['content'].replace('\n', ' ').replace('\t', ' ')))

    def collect_corpus(self, cmd='all'):
        f_p = open(self.corpus_path+'/company.pos', 'w+')
        f_n = open(self.corpus_path+'/company.neg', 'w+')
        for company in self.company_list:
            short_name = company['short_name']
            full_name = company['full_name']
            path = "%s/%s/%s" % (self.corpus_path, short_name, short_name)

            if cmd == 'match':
                docs = self.read_es(full_name)
                self.match(docs)
                self.remove_others(short_name) #删除关于其他公司的语料

            elif cmd == 'filter':
                data_neg, data_pos = self.filter(path+'_raw.txt')

                f_neg = open(path + '_neg.txt', 'wt')
                f_pos = open(path + '_pos.txt', 'wt')

                self.save(data_neg, f_neg)
                self.save(data_pos, f_pos)

            elif cmd == 'all':
                docs = self.read_es(full_name)
                self.match(docs)
                self.remove_others(short_name)  # 删除关于其他公司的语料

                f_neg = open(path + '_neg.txt', 'wt')
                f_pos = open(path + '_pos.txt', 'wt')

                data_neg, data_pos = self.filter(path + '_raw.txt')
                self.save(data_neg, f_neg)
                self.save(data_pos, f_pos)

            else:
                print("invalid command!")

            f_neg.close()
            f_pos.close()

            self.save(data_neg, f_n)
            self.save(data_pos, f_p)

        f_n.close()
        f_p.close()

    #输入公司全名  输出docs
    def read_es(self, company_name):
        query = {
            "query": {
                "bool": {
                    "filter": {
                        "bool": {
                            "must": [
                                {
                                    "term": {
                                        "company_name": "xxx"
                                    }
                                },
                                {
                                    "term": {
                                        "source": "news"
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        }
        query["query"]["bool"]["filter"]["bool"]["must"][0]["term"]["company_name"] = company_name

        esindex = es_index('192.168.2.46', 9200, 'online2_news_index', 'online_news')
        docs = esindex.search_index_scroll(query)
        print("doc_cnt:%d" % (len(docs)))
        # print(docs[0])
        return docs

    def save_docs(self, docs, path):
        for doc in docs:
            pass

    def remove_others(self, short_name):
        import os
        cmd = "cd %s; ls *.txt|grep -v %s_raw.txt|xargs rm"%(self.corpus_path, short_name)
        res = os.system(cmd)
        print("%s: %d"%(cmd, res))

        cmd = "cd %s; mkdir -p %s; mv %s_raw.txt ./%s/"%(self.corpus_path, short_name, short_name, short_name)
        res = os.system(cmd)
        print("%s: %d"%(cmd, res))

    def match(self,  docs, batch_size=2000):
        d = {'title': '万达商业获得1.6亿美元投资，王健林很愉快', 'content': '', 'filter': set(), 'filter_reason': set(),
             'url': '', 'publish_time': '2018-01-01 12:00:00', 'data_type': '', 'src': ''}
        data_res = []
        data_in = []
        for _d in docs:
            td = copy.deepcopy(d)
            td['title'] = _d['_source']['title']
            td['content'] = _d['_source']['content']
            data_in.append(td)
            if len(data_in) == batch_size:
                data_res += company_match(data_in)
                data_in = []
        data_res += company_match(data_in)
        # print("data_res: %s"%data_res[0])
        cnt1 = len(list(filter(lambda o: len(o['filter']) == 0, data_res)))
        cnt2 = len(list(filter(lambda o: len(o['filter']) >= 1, data_res)))
        print("OK_cnt:%d, Bad_cnt:%d" % (cnt1, cnt2))

    def filter(self, infile):
        data_pos, data_neg = [], []
        with open(infile, 'rt') as f_raw:
            for line in f_raw:
                items = line.strip().split('\t')
                if len(items) != 5 or line.find('acm res:') != 0:
                    continue
                if items[2] == '0':
                    data_neg.append([items[1],
                                     items[3].replace(items[1], self.COMPANY_NEG),
                                     items[4].replace(items[1], self.COMPANY_NEG)])
                else:
                    data_pos.append([items[1],
                                     items[3].replace(items[1], self.COMPANY_POS),
                                     items[4].replace(items[1], self.COMPANY_POS)])

        return data_neg, data_pos

    def connect(self):
        pass

    def save(self, corpus, f):
        for l in corpus:
            f.write('\t'.join(l)+'\n')

    def load(self, path):
        pass
