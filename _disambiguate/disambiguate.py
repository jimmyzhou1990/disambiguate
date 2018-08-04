import gensim
import jieba

class Disambiguate(object):
    def __init__(self, conf):
        jieba.load_userdict(conf['user_dict'])  # 加载自定义词典
        self.w2v_model = gensim.models.Word2Vec.load(conf['model_path'])
        self.vocab_set = set(self.w2v_model.wv.vocab)
        self.test_corpus_path = conf['test_corpus_path']
        self.COMPANY_NEG = conf['COMPANY_NEG']
        self.COMPANY_POS = conf['COMPANY_POS']
        self.cases = []
        self.window = int(conf[''])

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
                case['similarity'] = self.cal_each_similarity(case)
                #print(case)
                self.cases.append(case)

    def find_position(self, case):
        position_list = []

        for index, word in enumerate(case['word_list']):
            if word == case['short_name']:
                position_list.append(index)
        if len(position_list) == 0:
            print("warning...there is no keyword:[%s] in wordlist:  \n[%s]" % (case['short_name'], case['word_list']))
            position_list.append(-1)
        return position_list

    def similarity_weight(self, diff):


    def cal_each_similarity(self, case):
        similarity_list = []
        for w in case['word_list']:
            if w not in self.vocab_set or w == case['short_name']:
                similarity_list = .0
            else:
                similarity_list.append(self.w2v_model.similarity(w, self.COMPANY_POS))
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

    def test(self):
        self.load_test_corpus()
        for case in self.cases:
            sum_p, sum_n = self.cal_similarity(case)
            print("[%s]---%s"%(case['short_name'], case['sentence']))
            print("similarity with COMPANY_POS: %f, similarity with COMPANY_NEG: %f, diff: %f"%(sum_p, sum_n, sum_p-sum_n))
            print("---------------------------------------------------------------------------------")

    def