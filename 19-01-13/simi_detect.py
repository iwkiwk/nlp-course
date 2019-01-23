# 抄袭判定算法流程
# 1. 新闻预料处理，标记出新闻来源，来源为“新华社”或“新华网”的均视为新华社消息，标记为1，其他来源新闻标记为0
# 2. 语料分句及分词之后写入文件，用作后面计算句子向量
# 3. 训练样本与测试样本筛选，从新华社新闻中随机抽取90%样本作为训练样本，其余为测试样本，其他新闻中选取与新华社训练相同样本数参与模型训练
# 4. 由于TF-IDF矩阵维数大、训练速度慢，本程序使用Word2Vec及词频率加权计算句子向量，作模型的输入，W2V使用之前训练的Wiki数据模型，并增加新的新闻语料训练
# 5. 抄袭判定规则：如果一篇新闻中有超过30%的句子判定为抄袭，则认为这篇新闻抄袭了新华社的新闻
# 6. 模型质量判断：计算模型判定的精确率和召回率判断模型的质量
# 7. 调整模型训练参数，根据模型的精确率和召回率来优化模型

import random
import re
from collections import Counter
from collections import defaultdict

import jieba
import numpy as np
import pandas as pd
import xgboost as xgb
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Load trained w2v model
def get_w2v_model(filename):
    model = Word2Vec.load(filename)
    return model


# Remove special characters
def rm_spec(sent):
    ret = re.sub('[\\n\s+\.\!\/_,$%^*(+\"\')]+|[+——\-()?【】《》“”！，。？、~@#￥%……&*（）]+', '', sent)
    if ret:
        return ret
    return ''


# Split sentence
def split_sent(sent):
    sents = re.split('[。？！\n]', sent)
    ret = []
    for s in sents:
        sl = jieba.lcut(s)
        slt = []
        for item in sl:
            sr = rm_spec(item)
            if sr:
                slt.append(sr)
        if slt:
            ret.append(' '.join(slt))
    return ret


# Process news
def process_news_corpus(filename, news, lst):
    di = defaultdict(list)
    xna_lst = []
    count = 0
    with open(filename, 'w', encoding='utf-8') as fout:
        for i, item in enumerate(lst):
            sent = news['content'][item]
            sents = split_sent(sent)
            if sents:
                di[i].append(count)
                count += len(sents)
                di[i].append(count)
                if '新华社' in news['source'][item] or '新华网' in news['source'][item]:
                    xna_lst.append(i)
                fout.write('\n'.join(sents) + '\n')
    return di, xna_lst


# Add new corpus to train w2v model
def add_more_train_corpus(model, filename):
    sen_list = []
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            sen_list.append(line.split())
    model.train(sentences=sen_list, total_examples=len(sen_list), epochs=1)


# Get news sentences index list from a dict structure
def get_index_lst_from_dict(di, lst):
    ret = []
    for i in lst:
        start = di[i][0]
        end = di[i][1]
        for t in range(start, end):
            ret.append(t)
    return ret


def get_idx_lst_from_dict(di, idx):
    ret = []
    start = di[idx][0]
    end = di[idx][1]
    for t in range(start, end):
        ret.append(t)
    return ret


# Calculate word frequency
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


# Get sentence vector matrix
def get_sent_vec(model, get_wd_freq, corpus_file):
    a = 0.001
    col = model.wv.vector_size
    with open(corpus_file, 'r', encoding='utf-8') as fin:
        all_lines = fin.readlines()
        ret = np.zeros((len(all_lines), col))
        for idx, line in enumerate(all_lines):
            sent_vec = np.zeros(col)
            wd_lst = line.split()
            for wd in wd_lst:
                try:
                    pw = get_wd_freq(wd)
                    w = a / (a + pw)
                    sent_vec += w * np.array(model.wv[wd])
                except:
                    pass
            ret[idx] = sent_vec
        return ret


# Load sentence vector from saved file
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


def get_tfidf_mat(news, lst):
    corpus = []
    for i in lst:
        corpus.append(' '.join(jieba.lcut(news['content'][i])))
    vectorizer = TfidfVectorizer()
    ret = vectorizer.fit_transform(corpus)
    return ret


def get_remain_list(all_samples, part_samples):
    ret = []
    for i in all_samples:
        if i not in part_samples:
            ret.append(i)
    return ret


def lst2file(filename, lst):
    with open(filename, 'w') as fout:
        fout.write('\n'.join([str(i) for i in lst]) + '\n')


# Calculate model's precision and recall rate
def get_precision_and_recall(xna_test_res, otr_test_res):
    cc1 = Counter(xna_test_res)
    cc2 = Counter(otr_test_res)
    tp = cc1[1]
    fp = cc2[1]
    tn = cc2[0]
    fn = cc1[0]
    preci = tp / (tp + fp)
    recal = tp / (tp + fn)
    return preci, recal


def KNNClassifier(xTrain, yTrain):
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(xTrain, yTrain)
    return neigh


def SVMClassifier(xTrain, yTrain):
    clf = SVC(gamma='auto')
    clf.fit(xTrain, yTrain)
    return clf


def DTClassifier(xTrain, yTrain):
    clf = DecisionTreeClassifier()
    clf.fit(xTrain, yTrain)
    return clf


def RFClassifier(xTrain, yTrain):
    clf = RandomForestClassifier(n_estimators=10, max_depth=10)
    clf.fit(xTrain, yTrain)
    return clf


class XGBClassifier:
    def __init__(self, xTrain, yTrain):
        dtrain = xgb.DMatrix(xTrain, label=yTrain)
        params = {'objective': 'binary:logitraw', 'silent': 1, 'n_estimators': 1000,
                  'max_depth': 8}
        self.clf = xgb.train(params, dtrain)

    def predict(self, mat):
        test = xgb.DMatrix(mat)
        res = self.clf.predict(test)
        return res.astype(int)


# Classify main process
def main_func(classifier, mat, idx_dict, xna_trn_lst, otr_trn_lst, xna_test_lst, otr_test_lst):
    xna_trn_sent_lst = get_index_lst_from_dict(idx_dict, xna_trn_lst)
    otr_trn_sent_lst = get_index_lst_from_dict(idx_dict, otr_trn_lst)
    # print('XNA news amount for training:', len(xna_trn_sent_lst))
    # print('Other news amount for training:', len(otr_trn_sent_lst))

    X_train = mat[xna_trn_sent_lst + otr_trn_sent_lst, :]
    Y = np.array([0] * mat.shape[0])
    for i in xna_trn_sent_lst: Y[i] = 1
    y_train = Y[xna_trn_sent_lst + otr_trn_sent_lst]

    trainedModel = classifier(X_train, y_train)

    xna_test = []
    otr_test = []
    threshold = 0.3  # Similiar sentence amount over 30% is classified as plagiarized

    for xt in xna_test_lst:
        sent_lst = get_idx_lst_from_dict(idx_dict, xt)
        scores = []
        for si in sent_lst:
            sco = trainedModel.predict([mat[si]])[0]
            scores.append(sco)
        cc = Counter(scores)
        if cc[1] / len(sent_lst) > threshold:
            xna_test.append(1)
        else:
            xna_test.append(0)

    for ot in otr_test_lst:
        sent_lst = get_idx_lst_from_dict(idx_dict, ot)
        scores = []
        for si in sent_lst:
            sco = trainedModel.predict([mat[si]])[0]
            scores.append(sco)
        cc = Counter(scores)
        if cc[1] / len(sent_lst) > threshold:
            otr_test.append(1)
        else:
            otr_test.append(0)

    precision, recall = get_precision_and_recall(xna_test, otr_test)
    print(classifier.__name__ + ' precision {}, recall {}'.format(precision, recall))

    # lst2file('trn_result.txt', xna_test)
    # lst2file('otr_result.txt', otr_test)


# Import original corpus
news_df = pd.read_csv('sqlResult_1558435.csv', encoding='gb18030')

# Get news list which has source and content
lst_with_content = get_list_with_content(news_df)

# Build news index dict of corpus file, and filter XNA news
index_dict, xna_news_lst = process_news_corpus('news_corpus.txt', news_df, lst_with_content)
otr_news_lst = get_remain_list(list(index_dict.keys()), xna_news_lst)

# Write temp results to file
# with open('index_dict.txt', 'w') as fout:
#     for k in index_dict:
#         fout.write(str(k) + ':' + ','.join([str(i) for i in index_dict[k]]) + '\n')
#
# with open('xna_lst.txt', 'w') as fout:
#     fout.write('\n'.join([str(i) for i in xna_news_lst]))
#     fout.write('\n')

# Set 90% samples for training, 10% for testing
samples_n = int(len(xna_news_lst) * 0.9)

xna_samples_train = random.sample(xna_news_lst, samples_n)
xna_samples_test = get_remain_list(xna_news_lst, xna_samples_train)

otr_samples = random.sample(otr_news_lst, len(xna_news_lst))
otr_samples_train = random.sample(otr_samples, samples_n)
otr_samples_test = get_remain_list(otr_samples, otr_samples_train)

# f = open('train_lst.txt', 'w')
# f.write('\n'.join([str(i) for i in xna_train_lst]))
# f.close()

# Get tfidf vector
# tfidf method for large corpus has large memory footprint, better to replace it with w2v embedding
# tfidf_mat = get_tfidf_mat(news_df, lst_with_content).toarray()

w2v_model = get_w2v_model('wiki_w2v.model')
add_more_train_corpus(w2v_model, 'news_corpus.txt')
get_word_prob = word_freq('news_corpus.txt')
all_sent_mat = get_sent_vec(w2v_model, get_word_prob, 'news_corpus.txt')

## KNN
main_func(KNNClassifier, all_sent_mat, index_dict, xna_samples_train, otr_samples_train,
          xna_samples_test, otr_samples_test)

## SVM
main_func(SVMClassifier, all_sent_mat, index_dict, xna_samples_train, otr_samples_train,
          xna_samples_test, otr_samples_test)

## Decision tree
main_func(DTClassifier, all_sent_mat, index_dict, xna_samples_train, otr_samples_train,
          xna_samples_test, otr_samples_test)

## Random forest
main_func(RFClassifier, all_sent_mat, index_dict, xna_samples_train, otr_samples_train,
          xna_samples_test, otr_samples_test)

## XGBoost
main_func(XGBClassifier, all_sent_mat, index_dict, xna_samples_train, otr_samples_train,
          xna_samples_test, otr_samples_test)
