import os
import re
from functools import reduce
from operator import and_

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def get_html_files():
    all_files = os.listdir('.')
    ret = list(filter(lambda p: 'html' in p, all_files))
    return ret


def get_word_id(vcter_, w):
    return vcter_.vocabulary_.get(w)


def get_sent_ids(vcter_, ws):
    ret = []
    for w in ws:
        id = get_word_id(vcter_, w)
        if id:
            ret.append(id)
    return ret


def search_docs(trs, sids):
    ret = reduce(and_, [set(np.where(trs[id])[0]) for id in sids])
    return ret


def get_cos_similarity(v1, v2):
    if len(v1) != len(v2):
        return 0
    return np.sum(np.array(v1) * np.array(v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_sorted_doc_ids(tfidf_array, doc_ids, sent_vec):
    ll = []
    for id in doc_ids:
        docvec = tfidf_array[id]
        dis = get_cos_similarity(docvec, sent_vec)
        ll.append((id, dis))

    sortedll = sorted(ll, key=lambda p: p[1], reverse=True)
    ret = [x[0] for x in sortedll]
    return ret


def get_replace_pat(ws):
    match = '(' + '|'.join(ws) + ')'
    pat = re.compile(match)
    return pat


def build_engine(htmls):
    corpus = [' '.join(jieba.lcut(open(f, 'r', encoding='gb18030').read())) for f in htmls]
    vectorizer = TfidfVectorizer()
    tfidf_array = vectorizer.fit_transform(corpus).toarray()
    trans_array = tfidf_array.T

    def core(query):
        words = jieba.lcut(query)
        sent_ids = get_sent_ids(vectorizer, words)
        doc_ids = search_docs(trans_array, sent_ids)
        sent_array = vectorizer.transform([' '.join(words)]).toarray()[0]
        sorted_doc_ids = get_sorted_doc_ids(tfidf_array, doc_ids, sent_array)

        for id in sorted_doc_ids:
            out = get_replace_pat(words).sub(repl='**\g<1>**', string=corpus[id])
            yield ''.join(out.split(' '))

    return core


if __name__ == '__main__':
    html_files = get_html_files()

    search_engine = build_engine(html_files)

    with open('results.md', 'w', encoding='utf-8') as f:
        for i, c in enumerate(search_engine('加工材料强度')):
            f.write('------------------------\n')
            f.write('result {}\n'.format(i + 1))
            f.write(c)
            f.write('\n')
            f.write('------------------------\n')
