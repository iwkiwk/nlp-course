import random
import re
from collections import Counter

import jieba
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier


def get_w2v_model(filename):
    model = Word2Vec.load(filename)
    return model


def rm_spec(sent):
    ret = re.sub('[\n\s+\.\!\/_,$%^*(+\"\')]+|[+——\-()?【】《》“”！，。？、~@#￥%……&*（）]+', '', sent)
    if ret:
        return ret
    return ''


def write_news_corpus_to_file(filename, news, lst):
    ret = []
    with open(filename, 'w', encoding='utf-8') as fout:
        for i in lst:
            sent = news['content'][i]
            sent = rm_spec(sent)
            wd_lst = jieba.lcut(sent)
            if wd_lst:
                fout.write(' '.join(wd_lst))
                fout.write('\n')
                ret.append(i)
    return ret


def add_more_train_corpus(model, filename):
    sen_list = []
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            sen_list.append(line.split())
    model.train(sentences=sen_list, total_examples=len(sen_list), epochs=1)


def word_freq(corpus_file):
    word_list = []
    with open(corpus_file, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            word_list += line.split()
    cc = Counter(word_list)
    num_all = sum(cc.values())

    def get_word_freq(word):
        return cc[word] / num_all

    return get_word_freq


def write_sent_vec_to_file(filename, model, get_wd_freq, corpus_file):
    a = 0.001
    row = model.wv.vector_size
    with open(filename, 'w') as fout:
        with open(corpus_file, 'r', encoding='utf-8') as fin:
            for line in fin.readlines():
                sent_vec = np.zeros(row)
                wd_lst = line.split()
                for wd in wd_lst:
                    try:
                        pw = get_wd_freq(wd)
                        w = a / (a + pw)
                        sent_vec += w * np.array(model.wv[wd])
                    except:
                        pass
                fout.write(' '.join([str(i) for i in sent_vec]))
                fout.write('\n')


def get_all_sent_vec(filename):
    ret = np.fromfile(filename, dtype=np.float, sep=' ')
    return np.reshape(ret, (-1, 100))


def get_list_with_content(news):
    ret = []
    for i in range(len(news)):
        if pd.isna(news['source'][i]) or pd.isna(news['content'][i]):
            continue
        ret.append(i)
    return ret


def get_xna_news_index(news, lst):
    ret = []
    for i in range(len(lst)):
        if '新华社' in news['source'][lst[i]] or '新华网' in news['source'][lst[i]]:
            ret.append(i)
    return ret


def get_tfidf_mat(news, lst):
    corpus = []
    for i in lst:
        corpus.append(' '.join(jieba.lcut(news['content'][i])))
    vectorizer = TfidfVectorizer()
    ret = vectorizer.fit_transform(corpus)
    return ret


def get_remain_samples_list(all_samples, part_samples):
    ret = []
    for i in all_samples:
        if i not in part_samples:
            ret.append(i)
    return ret


# Classify using kNN
def KNNClassifier(mat, xna_trn_lst, otr_trn_lst, xna_test_lst, otr_test_lst):
    X_train = mat[xna_trn_lst + otr_trn_lst, :]
    y_train = np.array([0] * X_train.shape[0])
    for i in xna_trn_lst:
        y_train[i] = 1

    neigh = KNeighborsClassifier(n_neighbors=20)
    neigh.fit(X_train, y_train)
    for i in range(10):
        print(mat[xna_test_lst[i]])
        print(neigh.predict([mat[xna_test_lst[i]]]))


news_df = pd.read_csv('sqlResult_1558435.csv', encoding='gb18030')
lst_with_content = get_list_with_content(news_df)
lst_with_content = write_news_corpus_to_file('news_corpus.txt', news_df, lst_with_content)

# Find XNA news in news corpus
xna_news_lst = get_xna_news_index(news_df, lst_with_content)

# Set 90% samples for training, 10% for testing
samples_n = int(len(xna_news_lst) * 0.9)
xna_samples_train = random.sample([i for i in range(len(xna_news_lst))], samples_n)
xna_samples_test = get_remain_samples_list(xna_news_lst, xna_samples_train)
other_samples_n = int((len(lst_with_content) - len(xna_news_lst)) * 0.9)
other_samples = get_remain_samples_list([i for i in range(len(lst_with_content))], xna_news_lst)
other_samples_train = random.sample(other_samples, other_samples_n)
other_samples_test = get_remain_samples_list(other_samples, other_samples_train)

# Get tfidf vector
# tfidf method for large corpus has large memory footprint, need to replace it with w2v embedding
# tfidf_mat = get_tfidf_mat(news_df, lst_with_content).toarray()

w2v_model = get_w2v_model('wiki_w2v.model')
add_more_train_corpus(w2v_model, 'news_corpus.txt')
get_word_prob = word_freq('news_corpus.txt')
write_sent_vec_to_file('sent_vec.txt', w2v_model, get_word_prob, 'news_corpus.txt')
all_sent_mat = get_all_sent_vec('sent_vec.txt')

KNNClassifier(all_sent_mat, xna_samples_train, other_samples_train, xna_samples_test, other_samples_test)
