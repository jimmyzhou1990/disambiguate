import numpy as np


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

def get_test_data(test_path, w2vec, window):
    x_test = []
    y_test = []
    vocab_set = set(w2vec.wv.vocab)

    with open(test_path, 'r') as f:
        for l in f:
            wordsplits = l.strip().split(" ")
            wordlist = wordsplits[0:window*2]
            label = [1, 0] if wordsplits[-1] == '1' else [0, 1]
            veclist = [w2vec[w] if w in vocab_set else np.zero(window*2) for w in wordlist]
            x_test.append(veclist)
            y_test.append(label)

    return np.array(x_test), np.array(y_test)
