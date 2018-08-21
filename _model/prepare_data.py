import numpy as np
import jieba
import gensim


def filter_word(word):
    import re
    if word == 'COMPANY_NAME' or word == 'COMPANY_POS' or word == 'COMPANY_NEG':
        return False

    pattern_str = '''[0-9]|[a-z]|[A-Z]|月|年|日|中|】|【|前|后|上午|
    再|原|一个|不断|时间|时|记者|获悉|.*网|报道|―|全国|相关|新|正式|全|本报讯|
    一|以来|称|上海|深圳|广州|重庆|北京|苏州|南京|杭州|武汉|江苏|国际|刚刚|查看|
    已|今天|近期|有望|一直|继续|昨天|五|预计'''

    pattern = re.compile(pattern_str)
    res = pattern.search(word)

    if res:
        return False
    return True


def load_sentence_feature(corpus_path, range, seq_length,  keyword, w2vec, vocab_set):
    x_set = []
    x_info = []
    with open(corpus_path, 'r') as f:
        for l in f:
            items = l.strip().split("\t")
            wordlist = items[1].split(" ")
            shortname = items[0]

            if  keyword not in wordlist:
                continue
            keyword_position = wordlist.index(keyword)

            veclist = []
            feature_wlist = []
            for index, w in enumerate(wordlist):
                if index < keyword_position - range or index > keyword_position + range:
                    continue

                if w in vocab_set and filter_word(w):
                    veclist.append(w2vec[w])
                    feature_wlist.append(w)
            if len(veclist) < seq_length:   #padding 0
                padding = [np.zeros(100)]*(seq_length-len(veclist))
                veclist = veclist + padding

            if len(feature_wlist) >= 2:
                x_set.append(veclist)
                x_info.append((shortname, "".join(wordlist), feature_wlist))
    return x_set, x_info


def get_lstm_dataset(conf):
    version = conf['lstm']['version']
    jieba.load_userdict(conf['user_dict'])  # 加载自定义词典
    w2vec = gensim.models.Word2Vec.load(conf['w2v_model_path']+'v1'+'/company_pos.w2vec')
    vocab_set = set(w2vec.wv.vocab)

    range = int(conf['lstm']['range'])
    corpus_path = conf['lstm']['corpus_path']
    company_neg = conf['COMPANY_NEG']
    company_pos = conf['COMPANY_POS']

    x_neg, x_neg_info = load_sentence_feature(corpus_path + version+'/lstm_title.neg',
                                     range, 2*range,  company_neg, w2vec, vocab_set)
    print(x_neg[0][0])
    print(x_neg[0][-1])
    y_neg = [[0, 1]] * len(x_neg)
    print("neg sample: %d"%len(x_neg))

    x_pos, x_pos_info = load_sentence_feature(corpus_path+ version +'/lstm_title.pos',
                                             range, 2*range, company_pos, w2vec, vocab_set)
    y_pos = [[1, 0]]*len(x_pos)
    print("pos sample: %d"%len(x_pos))

    x_set = x_neg + x_pos
    y_set = y_neg + y_pos
    x_info = x_neg_info + x_pos_info

    import random
    randnum = 50   #固定训练集和测试集
    random.seed(randnum)
    random.shuffle(x_set)
    random.seed(randnum)
    random.shuffle(y_set)
    random.seed(randnum)
    random.shuffle(x_info)

    train_set_len = int(len(x_set) * 0.8)
    x_train = np.array(x_set[0:train_set_len])
    y_train = np.array(y_set[0:train_set_len])
    x_test = np.array(x_set[train_set_len:])
    y_test = np.array(y_set[train_set_len:])
    x_test_info = x_info[train_set_len:]

    print("training set 1-5:")
    print(y_train[0:5])
    print(x_train.shape)
    return x_train, y_train, x_test, y_test, x_test_info










