import time
from collections import defaultdict

import jieba
import pandas as pd
from gensim.models import Word2Vec


def get_model_from_file(filename):
    model = Word2Vec.load(filename)
    return model


def get_related_words(word, model, max_size):
    start = [word]
    seen = defaultdict(int)

    while len(seen) < max_size:
        cur = start.pop(0)
        seen[cur] += 1

        for w, r in model.wv.most_similar(cur, topn=20):
            seen[w] += 1
            start.append(w)

    return seen


def write_news_corpus_to_file(news, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(len(news)):
            try:
                co = news['content'][i].replace('\r\n', '\n')
                for line in co.split('\n'):
                    l = line.strip()
                    if l:
                        li = jieba.lcut(l)
                        f.write(' '.join(li))
                        f.write('\n')
            except:
                print(news['content'][i])


def add_more_train_corpus(model, filename):
    sen_list = []
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            sen_list.append(line.split(' '))
    model.train(sentences=sen_list, total_examples=len(sen_list), epochs=1)


if __name__ == '__main__':
    w2v_model = get_model_from_file('wiki_w2v.model')
    w2v_model.wv.most_similar('说道', topn=100)

    news_df = pd.read_csv('sqlResult_1558435.csv', encoding='gb18030')
    # write_news_corpus_to_file(news_df, 'news_corpus.txt')

    start_time = time.time()
    add_more_train_corpus(w2v_model, 'news_corpus.txt')
    print('elapsed time: {}'.format(time.time() - start_time))
    w2v_model.wv.most_similar('说道', topn=100)

    related_words = get_related_words('说道', w2v_model, 500)
    related = sorted(related_words.items(), key=lambda x: x[1], reverse=True)

    similar_words = filter(lambda x: x[1] > 10, related)
    for w, c in similar_words:
        print('{}\t{}'.format(w, c))
