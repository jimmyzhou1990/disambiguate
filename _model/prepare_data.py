import numpy as np
import jieba
import gensim
import re


def filter_word(word, vocab):

    if word == 'COMPANY_NAME' or word == 'COMPANY_POS' or word == 'COMPANY_NEG':
        return ''

    if word not in vocab:
        return 'UnknownWord'

    pattern_str = '^\d{6}$'   #股票代码
    pattern = re.compile(pattern_str)
    res = pattern.match(word)
    if res:
        return word

    # pattern_str = '''[0-9]|[a-z]|[A-Z]|^[年月日中]$|】|【|前|后|上午|
    # 再|原|一个|不断|时间|时|记者|获悉|.*网|报道|―|全国|相关|新|正式|全|本报讯|
    # 一天|以来|称|刚刚|查看|
    # 已|今天|近期|有望|一直|继续|昨天|预计'''
    pattern_str = '\d+\.*\d*%*'

    pattern = re.compile(pattern_str)
    res = pattern.match(word)

    if res:
        return '8'

    return word

def extend_vector(shortname, vec):
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
        v3 = np.array([1., 0,  0, 0])
        vec = np.concatenate((vec, v3))
    else:
        v0 = np.array([0, 0, 0, 0])
        vec = np.concatenate((vec, v0))
    return vec

#preceding and succeeding
def load_sentence_feature(corpus_path, range, seq_length,  keyword, w2vec, vocab_set):
    x_set = []
    x_info = []
    with open(corpus_path, 'r') as f:
        for l in f:
            items = l.strip().split("\t")
            wordlist = items[1].split(" ")
            shortname = items[0]

            if keyword not in wordlist:
                continue
            keyword_position = wordlist.index(keyword)

            # preceding
            pre_veclist = []
            pre_wordlist = []
            for w in wordlist[keyword_position:0:-1]:
                w = filter_word(w, vocab_set)
                if w:
                    pre_wordlist.insert(0, w)
                    pre_veclist.insert(0, w2vec[w])
                if len(pre_wordlist) >= range:
                    break
            if len(pre_wordlist) < range: # padding 0
                padding = [np.zeros(100)] * (range - len(pre_wordlist))
                pre_veclist += padding

            #succeeding
            suc_veclist = []
            suc_wordlist = []
            for w in wordlist[keyword_position:]:
                w = filter_word(w, vocab_set)
                if w:
                    suc_wordlist.insert(0, w)
                    suc_veclist.insert(0, w2vec[w])
                if len(suc_wordlist) >= range:
                    break
            if len(suc_wordlist) < range: #padding 0
                padding = [np.zeros(100)]*(range-len(suc_wordlist))
                suc_veclist += padding

            #concat
            veclist = pre_veclist + suc_veclist
            info = (shortname, "".join(wordlist), pre_veclist+suc_wordlist)

            x_set.append(veclist)
            x_info.append(info)

    return x_set, x_info


def get_lstm_dataset(conf):
    version = conf['lstm']['version']
    jieba.load_userdict(conf['user_dict'])  # 加载自定义词典
    w2vec_version = conf['lstm']['w2vec_version']
    w2vec = gensim.models.Word2Vec.load(conf['w2v_model_path']+w2vec_version+'/company_pos.w2vec')
    vocab_set = set(w2vec.wv.vocab)

    range = int(conf['lstm']['range'])
    corpus_path = conf['lstm']['corpus_path']
    company_neg = conf['COMPANY_NEG']
    company_pos = conf['COMPANY_POS']

    x_neg, x_neg_info = load_sentence_feature(corpus_path + version+'/lstm_title.neg',
                                     range, 2*range,  company_neg, w2vec, vocab_set)
    print(x_neg[0])
 
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







