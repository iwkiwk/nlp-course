import os
from pyltp import Segmentor  # 分词
from pyltp import Postagger  # 词性标注
from pyltp import NamedEntityRecognizer  # 命名实体识别
from pyltp import Parser  # 句法解析

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
import math
import numpy as np
import jieba

import nltk
from nltk.parse import DependencyGraph
from nltk.tree import Tree

LTP_DATA_DIR = 'D:\\ltp_data_v3.4.0'
# sentence = '小明说：“这个苹果真好吃！”'
# sentence = '昨日，雷先生说，交警部门罚了他 16 次，他只认了一次，交了一次罚款，拿到法院的判决书后，会前往交警队，要求撤销此前的处罚。'
# sentence = '但韩国网友对“韩国海军陆战队世界第二”的说法不以为然。'
# sentence = '不少网友留言嘲讽称：“这似乎是韩国海军陆战队争取国防预算的软文”“现在很多韩国海军陆战队员都是戴眼镜、瘦豆芽体型，不知道怎么选拔的”“记者大概是海军陆战队退役的吧”。'
sentence = '《中央日报》称，当前韩国海军陆战队拥有2个师和2个旅，还打算在2021年增设航空团，并从今年开始引进30余架运输直升机和20架攻击直升机。此外，韩军正在研发新型登陆装甲车，比现有 AAV-7 的速度更快、火力更猛。未来韩国海军陆战队还会配备无人机，“将在东北亚三国中占据优势”。'


# 寻找依存树根节点编号
def get_dependtree_root_index(word_list):
    # 词性标注
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
    postagger = Postagger()
    postagger.load(pos_model_path)
    postags = postagger.postag(word_list)
    # print(list(postags))
    postagger.release()

    # 命名实体识别
    ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
    recognizer = NamedEntityRecognizer()
    recognizer.load(ner_model_path)
    netags = recognizer.recognize(word_list, postags)
    # print(list(netags))
    recognizer.release()

    # 句法依存关系
    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
    parser = Parser()
    parser.load(par_model_path)
    arcs = parser.parse(word_list, postags)
    # print(' '.join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
    parser.release()

    for i in range(len(arcs)):
        if arcs[i].head == 0:
            return i, postags, arcs  # 同时返回词性及依存关系列表
    return -1, postags, arcs


# 寻找依存关系子节点
def get_child_index(ind, arcs):
    ret = []
    for i in range(len(arcs)):
        if arcs[i].head == ind + 1:
            ret.append(i)

    return ret


# 获取命名实体索引
def get_ne_index(postags, chd_list):
    ret = []
    for i in chd_list:
        if postags[i] in ['n', 'nh', 'ni']:
            ret.append(i)
    return ret


# 获取中心词之后的第一个符号的索引
def get_first_wp_after_index(postags, after):
    for i in range(after + 1, len(postags)):
        if postags[i] == 'wp':
            return i
    return 0


# 获取句号索引列表
def get_periods_index_after(word_list, after):
    ret = []
    for i in range(after + 1, len(word_list)):
        if word_list[i] == '。':
            ret.append(i)
    return ret


# 获取长句中的分句，为下面的句子向量分析作准备
def get_sent_list(word_list, start, periods):
    ret = []
    for i, p in enumerate(periods):
        if i == 0:
            ret.append(list(word_list[start + 1:p + 1]))
        else:
            ret.append(list(word_list[periods[i - 1] + 1:p + 1]))
    return ret


# 获取语料库TF-IDF vectorizer
def get_tfidf_vectorizer(corpus_file):
    corpus = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            l = line.strip()
            if l:
                corpus.append(l)
            else:
                break

    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')  # 不过滤单汉字
    X = vectorizer.fit_transform(corpus)
    return vectorizer


# 获取句子向量
def get_sentence_vec(vectorizer, word_list):
    trans = vectorizer.transform([' '.join(word_list)])
    return trans.toarray()[0]


# 计算余弦相似度
def get_cos_similarity(v1, v2):
    if len(v1) != len(v2):
        return 0
    return np.sum(np.array(v1) * np.array(v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))


if __name__ == '__main__':
    # 分词
    # cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
    # segmentor = Segmentor()
    # segmentor.load(cws_model_path)
    # words = segmentor.segment(sentence)
    # print(list(words))
    # segmentor.release()

    words = jieba.lcut(sentence)
    print(words)

    # 获取中心词，词性列表，依存关系表
    root_index, postags, arcs = get_dependtree_root_index(words)
    print(words[root_index])

    wp_index = get_first_wp_after_index(postags, root_index)
    print(wp_index)

    periods_index = get_periods_index_after(words, wp_index)
    print(periods_index)

    # 分句
    sents = get_sent_list(words, wp_index, periods_index)
    for sen in sents:
        print(sen)

    # 获取完整命名实体，针对命名实体词被分割的情况
    children = get_child_index(root_index, arcs)
    print(children)

    ne_list = get_ne_index(postags, children)

    oth = []
    for ne in ne_list:
        nechd = get_child_index(ne, arcs)
        oth.append(get_ne_index(postags, nechd))

    for i, n in enumerate(ne_list):
        if oth[i]:
            print(words[oth[i][0]] + words[n])
        else:
            print(words[n])

    # 绘制依存树
    conll = ''
    for i in range(len(arcs)):
        if arcs[i].head == 0:
            arcs[i].relation = 'ROOT'
        conll += words[i] + '(' + postags[i] + ')' + '\t' + postags[i] + '\t' + str(arcs[i].head) + '\t' + arcs[
            i].relation + '\n'
    print(conll)

    conlltree = DependencyGraph(conll)
    tree = conlltree.tree()
    tree.draw()

    # 句子向量化
    vectorizer = get_tfidf_vectorizer('train_corpus2.txt')
    features = vectorizer.get_feature_names()

    vec = get_sentence_vec(vectorizer, ['今天', '天气', '不错'])
    # print(vec)

    vec3 = get_sentence_vec(vectorizer, ['今天', '天气', '还行'])
    vec4 = get_sentence_vec(vectorizer, ['今天', '天气', '不错'])
    # print(vec3)
    # print(vec4)
    print(get_cos_similarity(vec3, vec4))
