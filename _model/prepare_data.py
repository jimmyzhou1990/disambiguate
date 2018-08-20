import numpy as np
import jieba
import gensim

def get_dataset(in_path, out_path, length, window, company_name, w2vec):
    data_set = []
    f_out = open(out_path, 'w')
    vocab_set = set(w2vec.wv.vocab)
    print("length: %d, window: %d, name: %s"%(length, window, company_name))
    with open(in_path, 'r') as f_in:
        for l in f_in:
            wordlist = l.strip().split(" ")
            #print(wordlist)    
            #print(wordlist[length])
            if (len(wordlist) < length*2)  or (wordlist[length] != company_name):
                #print("pass...length: %d, mid: %s"%(len(wordlist), wordlist[length]))
                continue
            #print(wordlist)
            wordlist_l = list(filter(lambda o: o != company_name and o != '\u2002' and o in vocab_set, wordlist[0:length]))
            wordlist_r = list(filter(lambda o: o != company_name and o != '\u2002' and o in vocab_set, wordlist[length+1:]))

            if len(wordlist_l) < window or len(wordlist_r) < window:
                continue

            wordnear = wordlist_l[-window:] + wordlist_r[0:window]
            f_out.write(" ".join(wordnear)+"\n")
            veclist = [w2vec[w] for w in wordnear]
            data_set.append(veclist)

    f_out.close()

    training_set_len = int(len(data_set) * 0.8)
    training_set = np.array(data_set[0:training_set_len])
    testing_set = np.array(data_set[training_set_len:])
    label = [[1, 0] for _ in data_set]
    training_label = np.array(label[0:training_set_len])
    testing_label = np.array(label[training_set_len:])

    print("training dataset: %d, testing data set: %d"%(len(training_set), len(testing_set)))

    return data_set, training_set, testing_set, training_label, testing_label

def get_test_data(test_path, user_dict, w2vec, window):
    x_test = []
    y_test = []
    vocab_set = set(w2vec.wv.vocab)
    jieba.load_userdict(user_dict)  # 加载自定义词典

    def find_company(wordlist, short_name):
        for index, word in enumerate(wordlist):
            if word == short_name:
                return index
        return -1

    def enlarge(wordlist, maxlen):
        while len(wordlist) < maxlen:
            wordlist = wordlist*2

        return wordlist[:maxlen]

    with open(test_path, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            short_name = items[0]
            sentence = items[1]
            label = items[2]

            wordlist = list(filter(lambda o: o in vocab_set, list(jieba.cut(items[1]))))
            if len(wordlist) < 2*window:
                print("wordlist too short: %d"%len(wordlist))
                continue

            #print(wordlist)

            pos = find_company(wordlist, short_name)
            if pos < 0:
                print("cannot find the keyword: %s"%short_name)
                continue

            word_extract_l = wordlist[(0 if pos-window<0 else pos-window):pos]
            word_extract_r = wordlist[pos+1: pos+1+window]

            if len(word_extract_l) == 0 and len(word_extract_r) > 0:
                word_extract_l = word_extract_r.copy()
                word_extract_l.reverse()
            elif len(word_extract_l) > 0 and len(word_extract_r) == 0:
                word_extract_r = word_extract_l.copy()
                word_extract_r.reverse()
            elif len(word_extract_r) == 0 and len(word_extract_l) == 0:
                continue
            else:
                word_extract_l = enlarge(word_extract_l, window)
                word_extract_r = enlarge(word_extract_r, window)

            #print("left: %s"%word_extract_l)
            #print("right: %s"%word_extract_r)
            word_extract = word_extract_l + word_extract_r
            print(word_extract)
            print("----------------------------------------------")
            veclist = [w2vec[w] for w in word_extract]
            y = [1, 0] if label == "1" else [0, 1]
            x_test.append(veclist)
            y_test.append(y)

        return np.array(x_test), np.array(y_test)

def filter_word(word):
    import re
    if word == 'COMPANY_NAME' or word == 'COMPANY_POS' or word == 'COMPANY_NEG':
        return False

    pattern_str = '''[0-9]|[a-z]|[A-Z]|月|年|日|中|】|【|前|后|上午|
    再|原|一个|不断|时间|时|记者|获悉|.*网|报道|―|全国|相关|新|正式|全|本报讯|
    一|以来|称|上海|深圳|广州|重庆|北京|苏州|南京|杭州|武汉|江苏|国际|刚刚|查看|
    已|今天|近期|有望|一直|继续|昨天|五|预计''';
    #print("feature filter patterh: %s"%pattern_str)

    pattern = re.compile(pattern_str)
    res = pattern.search(word)

    if res:
        return False
    return True

def topn_similarity(companyword, keyword_position, wordlist, topn, range, w2vec, vocab_set):
    topn_simi_list = []
    topn_offset_list = []
    simi_list = []

    for index, w in enumerate(wordlist):
        if index < keyword_position-range or index > keyword_position+range:
            continue
        simi = w2vec.similarity(w, companyword) if w in vocab_set and filter_word(w) else -1
        offset = index-keyword_position if index>=keyword_position else keyword_position-index
        offset = offset/range
        simi_list.append((simi, offset, w))

    topn_sorted = sorted(simi_list, key=lambda x: x[0], reverse=True)
    #print(topn_sorted[0:topn])

    for offset, simi, _ in topn_sorted[0:topn]:
        topn_simi_list.append(simi)
        topn_offset_list.append(offset)

    return topn_offset_list, topn_simi_list, simi_list, topn_sorted[0:topn]

def load_feature_set(corpus_path, window, range, topn, keyword, w2vec, vocab_set):
    x_set = []
    x_info = []
    with open(corpus_path, 'r') as f:
        for l in f:
            items = l.strip().split("\t")
            wordlist = items[1].split(" ")
            shortname = items[0]


            if len(wordlist) < window or keyword not in wordlist:
                continue

            keyword_position = wordlist.index(keyword)

            topn_offset_list, topn_simi_list, simi_list, top_list = topn_similarity("COMPANY_NAME", keyword_position, wordlist,
                                                               topn, range, w2vec, vocab_set)

            feature = topn_simi_list + topn_offset_list  # [offset/15 for offset in topn_offset_list]
            x_set.append(feature)
            x_info.append((shortname, "".join(wordlist), simi_list, top_list))

    return x_set, x_info

def load_feature(corpus_path, window, range, topn, keyword, w2vec, vocab_set):
    x_set = []
    x_info = []
    with open(corpus_path, 'r') as f:
        for l in f:
            items = l.strip().split("\t")
            wordlist = items[1].split(" ")
            shortname = items[0]

            if len(wordlist) < window or keyword not in wordlist:
                continue
            keyword_position = wordlist.index(keyword)

            vecsum = np.zeros(100)
            feature_wlist = []
            for index, w in enumerate(wordlist):
                if index < keyword_position - range or index > keyword_position + range:
                    continue

                if w in vocab_set and filter_word(w):
                    vecsum += w2vec[w]
                    feature_wlist.append(w)

            x_set.append(vecsum)
            x_info.append((shortname, "".join(wordlist), feature_wlist))
    return x_set, x_info

def get_lr_model_dataset(conf):
    jieba.load_userdict(conf['user_dict'])  # 加载自定义词典
    w2vec = gensim.models.Word2Vec.load(conf['w2v_model_path'])
    vocab_set = set(w2vec.wv.vocab)

    window = int(conf['lr']['window'])
    range = int(conf['lr']['range'])
    topn = int(conf['lr']['topn'])
    corpus_path = conf['lr']['corpus_path']
    company_neg = conf['COMPANY_NEG']
    company_pos = conf['COMPANY_POS']


    x_neg, x_neg_info = load_feature(corpus_path+'/extract_%d_lr_cut.neg'%window,
                                             window, range, topn, company_neg, w2vec, vocab_set)
    y_neg = [0]*len(x_neg)
    print("neg sample: %d"%len(x_neg))

    x_pos, x_pos_info = load_feature(corpus_path+'/extract_%d_lr_cut.pos'%window,
                                             window, range, topn, company_pos, w2vec, vocab_set)
    y_pos = [1]*len(x_pos)
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
    print(x_train[0:5])
    print(y_train[0:5])

    return x_train, y_train, x_test, y_test, x_test_info


def load_sentence_feature(corpus_path, window, range, seq_length,  keyword, w2vec, vocab_set):
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

            if len(feature_wlist) >= 1:
                x_set.append(veclist)
                x_info.append((shortname, "".join(wordlist), feature_wlist))
    return x_set, x_info


def get_lstm_dataset(conf):
    jieba.load_userdict(conf['user_dict'])  # 加载自定义词典
    w2vec = gensim.models.Word2Vec.load(conf['w2v_model_path'])
    vocab_set = set(w2vec.wv.vocab)

    window = int(conf['lr']['window'])
    range = int(conf['lr']['range'])
    topn = int(conf['lr']['topn'])
    corpus_path = conf['lstm']['corpus_path']
    company_neg = conf['COMPANY_NEG']
    company_pos = conf['COMPANY_POS']

    x_neg, x_neg_info = load_sentence_feature(corpus_path + '/lstm_title.neg',
                                     window, range, 2*range,  company_neg, w2vec, vocab_set)
    print(x_neg[0][0])
    print(x_neg[0][-1])
    y_neg = [[0, 1]] * len(x_neg)
    print("neg sample: %d"%len(x_neg))

    x_pos, x_pos_info = load_sentence_feature(corpus_path+'/lstm_title.pos',
                                             window, range, 2*range, company_pos, w2vec, vocab_set)
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
    #print(x_train[0:5])
    print(y_train[0:5])
    print(x_train.shape)
    return x_train, y_train, x_test, y_test, x_test_info










