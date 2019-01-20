import random

import jieba
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier


def get_w2v_model(filename):
    model = Word2Vec.load(filename)
    return model


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


news_df = pd.read_csv('sqlResult_1558435.csv', encoding='gb18030')
lst_with_content = get_list_with_content(news_df)

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
tfidf_mat = get_tfidf_mat(news_df, lst_with_content).toarray()

X_train = tfidf_mat[xna_samples_train + other_samples_train, :]
y_train = np.array([0] * X_train.shape[0])
for i in xna_samples_train:
    y_train[i] = 1

# Classify using kNN
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_train, y_train)
for i in range(10):
    print(neigh.predict([tfidf_mat[xna_samples_test[i]]]))
