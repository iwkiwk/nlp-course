import os
from pyltp import Segmentor  # 分词
from pyltp import Postagger  # 词性标注
from pyltp import NamedEntityRecognizer  # 命名实体识别
from pyltp import Parser  # 句法解析

import nltk
from nltk.parse import DependencyGraph
from nltk.tree import Tree

LTP_DATA_DIR = 'D:\\ltp_data_v3.4.0'
# sentence = '小明说：“这个苹果真好吃！”'
# sentence = '昨日，雷先生说，交警部门罚了他 16 次，他只认了一次，交了一次罚款，拿到法院的判决书后，会前往交警队，要求撤销此前的处罚。'
sentence = '《中央日报》称，当前韩国海军陆战队拥有2个师和2个旅，还打算在2021年增设航空团，并从今年开始引进30余架运输直升机和20架攻击直升机。'


# 寻找依存树根节点编号
def get_dependtree_root_index(word_list):
    # 词性标注
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
    postagger = Postagger()
    postagger.load(pos_model_path)
    postags = postagger.postag(word_list)
    print(list(postags))
    postagger.release()

    # 命名实体识别
    ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
    recognizer = NamedEntityRecognizer()
    recognizer.load(ner_model_path)
    netags = recognizer.recognize(word_list, postags)
    print(list(netags))
    recognizer.release()

    # 句法依存关系
    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
    parser = Parser()
    parser.load(par_model_path)
    arcs = parser.parse(word_list, postags)
    print(' '.join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
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


if __name__ == '__main__':
    # 分词
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
    segmentor = Segmentor()
    segmentor.load(cws_model_path)
    words = segmentor.segment(sentence)
    print(list(words))
    segmentor.release()

    # words = sentence.split(' ')
    # print(words)

    root_index, postags, arcs = get_dependtree_root_index(words)
    print(words[root_index])

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
