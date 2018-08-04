class CorpusCleaner(object):
    def __init__(self, conf):
        self.keyword_neg_path = conf['clean_config']['keyword_neg']
        self.keyword_pos_path = conf['clean_config']['keyword_pos']
        self.corpus_in_path = conf['clean_config']['corpus_in']
        self.corpus_out_path = conf['clean_config']['corpus_out']
        self.corpus = []
        self.keywords_pos = []
        self.keywords_neg = []

        with open(self.keyword_pos_path, 'rt') as f:
            for l in f:
                key = l.strip()
                self.keywords_pos.append(key)

        with open(self.keyword_neg_path, 'rt') as f:
            for l in f:
                key = l.strip()
                self.keywords_neg.append(key)

        print(self.keywords_pos)
        print(self.keywords_neg)

    def clean(self):
        f_o = open(self.corpus_out_path, 'a+')

        with open(self.corpus_in_path, 'rt') as f:
            for l in f:
                items = l.strip().split('\t')
                if len(items) != 5 or l.find('acm res:') != 0:
                    continue

                s = items[3] + " " + items[4]
                #print(type(s))
                if items[2] == '0' and \
                    self.find_keyword(s, self.keywords_pos) == True and \
                        self.find_keyword(s, self.keywords_neg) == False:
                    items[2] = '1'

                l = '\t'.join(items)+'\n'
                f_o.write(l)

        f_o.close()

    def find_keyword(self, s, key_words):
        for key in key_words:
            if s.find(key) >= 0:
                return True
        return False
