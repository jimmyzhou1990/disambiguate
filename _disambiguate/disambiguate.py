import gensim
import jieba

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
            similarity_list_offset.append((index, w, weight * simi))

        return  similarity_list_offset

    def summary_similarity(self, similarity_list_final):
        sum = .0
        for index, _, simi in similarity_list_final:
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
            print('topn similarity:')
            print(similarity_list_topn)
            similarity_list_final = self.cal_similarity_with_offset(similarity_list_topn, position)

            similarity_summary = self.summary_similarity(similarity_list_final)

            print("[%s]---%s"%(case['short_name'], case['sentence']))
            print("similarity with COMPANY_POS: %f"%similarity_summary)
            print("---------------------------------------------------------------------------------")
