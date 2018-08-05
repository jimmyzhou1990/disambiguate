import numpy as np
import jieba

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

            wordlist = filter(lambda o: o in vocab_set, list(jieba.cut(items[1])))
            if len(wordlist) < 2*window:
                print("wordlist too short: %d"%len(wordlist))
                continue

            print(wordlist)

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

            print("left: %s"%word_extract_l)
            print("right: %s"%word_extract_r)
            word_extract = word_extract_l + word_extract_r

            veclist = [w2vec[w] for w in word_extract]
            y = [1, 0] if label == "1" else [0, 1]
            x_test.append(veclist)
            y_test.append(y)

        return np.array(x_test), np.array(y_test)
            
