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


def topn_similarity(keyword, wordlist, topn, window, w2vec, vocab_set):
    topn_simi_list = []
    topn_offset_list = []
    topn_list = []

    for index, w in enumerate(wordlist):
        simi = w2vec.similarity(w, keyword) if w in vocab_set and w != '\u2002' else 0
        offset = index-window if index>=window else window-index
        topn_list.append((offset, simi, w))

    topn_sorted = sorted(topn_list, key=lambda x: x[1], reverse=True)
    #print(topn_sorted[0:topn])

    for offset, simi, _ in topn_sorted[0:topn]:
        topn_simi_list.append(simi)
        topn_offset_list.append(offset)

    return topn_offset_list, topn_simi_list

def load_feature_set(corpus_path, window, topn, keyword, w2vec, vocab_set):
    x_set = []

    with open(corpus_path, 'r') as f:
        for l in f:
            wordlist = l.strip().split(" ")
            if (len(wordlist) < window*2)  or (wordlist[window] != keyword):
                continue
            #print(wordlist)
            wordlist_l = list(filter(lambda o: o != keyword and o != '\u2002' and o in vocab_set,
                                     wordlist[0:window]))
            wordlist_r = list(filter(lambda o: o != keyword and o != '\u2002' and o in vocab_set,
                                     wordlist[window+1:]))

            if len(wordlist_l) < topn or len(wordlist_r) < topn:
                continue

            topn_offset_list, topn_simi_list = topn_similarity("COMPANY_NAME", wordlist,
                                                               topn, window, w2vec, vocab_set)

            feature = topn_simi_list + topn_offset_list
            x_set.append(feature)

    return x_set

def get_lr_model_dataset(conf):
    jieba.load_userdict(conf['user_dict'])  # 加载自定义词典
    w2vec = gensim.models.Word2Vec.load(conf['w2v_model_path'])
    vocab_set = set(w2vec.wv.vocab)

    window = conf['lr']['window']
    topn = conf['lr']['topn']
    corpus_path = conf['lr']['corpus_path']
    company_neg = conf['COMPANY_NEG']
    company_pos = conf['COMPANY_POS']

    x_neg = load_feature_set(corpus_path+'/extract_20_lr_cut.neg',
                                             window, topn, company_neg, w2vec, vocab_set)
    y_neg = [0]*len(x_neg)
    print("neg sample: %d"%len(x_neg))

    x_pos = load_feature_set(corpus_path+'/extract_20_lr_cut.pos',
                                             window, topn, company_pos, w2vec, vocab_set)
    y_pos = [1]*len(x_pos)
    print("pos sample: %d"%len(x_pos))

    x_set = x_neg + x_pos
    y_set = y_neg + y_pos

    import random
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(x_set)
    random.seed(randnum)
    random.shuffle(y_set)

    train_set_len = int(len(x_set) * 0.8)
    x_train = np.array(x_set[0:train_set_len])
    y_train = np.array(y_set[0:train_set_len])
    x_test = np.array(x_set[train_set_len:])
    y_test = np.array(y_set[train_set_len:])

    print(x_train[0:5])
    print(y_train[0:5])

    return x_train, y_train, x_test, y_test











