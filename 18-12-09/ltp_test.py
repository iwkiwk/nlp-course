import os
from pyltp import Segmentor  # 分词
from pyltp import Postagger  # 词性标注
from pyltp import NamedEntityRecognizer  # 命名实体识别
from pyltp import Parser  # 句法解析

import nltk
from nltk.parse import DependencyGraph
from nltk.tree import Tree

LTP_DATA_DIR = 'D:\\ltp_data_v3.4.0'
sentence = '元芳你怎么看？'

if __name__ == '__main__':
    # 分词
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
    segmentor = Segmentor()
    segmentor.load(cws_model_path)
    words = segmentor.segment(sentence)
    print(list(words))
    segmentor.release()

    # 词性标注
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
    postagger = Postagger()
    postagger.load(pos_model_path)
    postags = postagger.postag(words)
    print(list(postags))
    postagger.release()

    # 命名实体识别
    ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
    recognizer = NamedEntityRecognizer()
    recognizer.load(ner_model_path)
    netags = recognizer.recognize(words, postags)
    print(list(netags))
    recognizer.release()

    # 句法依存关系
    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
    parser = Parser()
    parser.load(par_model_path)
    arcs = parser.parse(words, postags)
    print(' '.join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
    parser.release()

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
