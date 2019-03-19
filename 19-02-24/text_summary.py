import re
from collections import Counter
from functools import partial

import jieba
import networkx as nx
import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import FastText
from gensim.models import LdaModel
from gensim.models.word2vec import LineSentence
from sklearn.metrics.pairwise import cosine_similarity


# generate corpus for fasttext use, only once
def generate_corpus_file(news_data):
    with open('corpus.txt', 'w', encoding='utf-8') as fout:
        for sent in news_data['content']:
            if pd.isna(sent):
                continue
            sent = re.sub(r'[\r\n]', '', sent)
            wl = jieba.lcut(sent)
            if wl:
                fout.write(' '.join(wl))
                fout.write('\n')


# split sentence, return sub sentence and sentence indices and punctuations indices
# punctuation indices for rebuilding complete sentence
def split_sentence(sent: str):
    sent = re.sub(r'[\r\n]', '', sent)
    ls = re.split('([，。？！,.])', sent)
    pat = '，。？！,.'
    sent_ids = []
    symbol_ids = []
    for i, s in enumerate(ls):
        if s in pat:
            symbol_ids.append(i)
        else:
            sent_ids.append(i)
    return ls, sent_ids, symbol_ids


#### Method 1: Using text rank ####
# build sub sentence connect graph
def get_connect_graph_by_text_rank(tokenized_text, window=3):
    keywords_graph = nx.Graph()
    tokeners = tokenized_text
    # print(tokeners)
    for ii, t in enumerate(tokeners):
        word_tuples = [(tokeners[connect], t)
                       for connect in range(ii - window, ii + window + 1)
                       if connect >= 0 and connect < len(tokeners)]
        keywords_graph.add_edges_from(word_tuples)

    return keywords_graph


def get_summarization_simple_with_text_rank(text, constraint=200):
    return get_summarization_simple(text, sentence_ranking_by_text_ranking, constraint)


# rank sub sentences
def sentence_ranking_by_text_ranking(split_sentence):
    sentence_graph = get_connect_graph_by_text_rank(split_sentence)
    ranking_sentence = nx.pagerank(sentence_graph)
    ranking_sentence = sorted(ranking_sentence.items(), key=lambda x: x[1], reverse=True)
    # print(ranking_sentence)
    return ranking_sentence


# main process
def get_summarization_simple(text, score_fn, constraint=200):
    sents, sent_ids, symb_ids = split_sentence(text)
    sub_sentence = [sents[i] for i in sent_ids]
    ranking_sentence = score_fn(sub_sentence)
    selected_text = set()
    current_text = ''

    for sen, _ in ranking_sentence:
        if len(current_text) < constraint:
            current_text += sen
            selected_text.add(sen)
        else:
            break

    summarized = []
    for sen in sub_sentence:  # print the selected sentence by sequent
        if sen in selected_text:
            ind = sents.index(sen)
            summarized.append(sen)
            if (ind + 1) in symb_ids:
                summarized.append(sents[ind + 1])
    return ''.join(summarized)


#### Method 2: using sentence embedding ####
# return word prob func
def get_prob(counter):
    total_cnt = sum(counter.values())

    def core(word: str):
        return counter[word] / total_cnt

    return core


# using stop words
def get_stop_words(file: str, encoding='utf-8'):
    ret = [l for l in open(file, encoding=encoding).read()]
    return ret


# calculate sentence vector, using average weighted word vector
def sentence_embedding(model, prob_func, stop_words):
    a = 0.001
    col = model.wv.vector_size

    def core(sent):
        vec = np.zeros(col)
        words = jieba.lcut(sent)
        for w in words:
            if w in model.wv.vocab and w not in stop_words:
                pw = a / (a + prob_func(w))
                vec += pw * model.wv[w]
        return vec

    return core


# rank cosine distance between sentence vector and whole text vector
def get_correlations(sents, sent_vec_func, text_vec=None):
    if text_vec is None:
        text = ' '.join(sents)
        text_vec = sent_vec_func(text)
    sims = []
    for sent in sents:
        vec = sent_vec_func(sent)
        sim = cosine_similarity(vec.reshape(1, -1), text_vec.reshape(1, -1))[0][0]
        sims.append(sim)
    ret = [(sents[ind], sims[ind]) for ind in sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)]
    return ret


#### Method 3: using LDA model ####
# similarity ranking using distance between sentence vec and topic vec
def get_topic_words(sent, stop_words, cnt=15):
    sent = re.sub(r'[\r\n]', '', sent)
    wlst = jieba.lcut(sent)
    ls = []
    for w in wlst:
        if w not in stop_words:
            ls.append(w)

    di = Dictionary([ls])
    corpus = [di.doc2bow(text) for text in [ls]]
    lda = LdaModel(corpus, id2word=di, num_topics=1)
    tp = lda.print_topics(num_words=cnt)[0][1]
    return re.findall('"(.+?)"', tp)


if __name__ == '__main__':
    news_data = pd.read_csv('../../input/sqlResult_1558435.csv', encoding='gb18030')
    news = news_data['content'][43]

    # generate corpus file only once
    # generate_corpus_file(news_data)

    ## method 1
    output = get_summarization_simple_with_text_rank(news, 250)
    print(output)

    ## method 2
    model = FastText(LineSentence('corpus.txt'), window=5, size=35, iter=10, min_count=1)
    tokeners = []
    for line in open('corpus.txt', 'r', encoding='utf-8'):
        tokeners += line.split()
    word_counter = Counter(tokeners)
    prob_func = get_prob(word_counter)

    stop_words = get_stop_words('../../input/stop_words.txt')
    get_sentence_vec = sentence_embedding(model, prob_func, stop_words)
    score_func_embed = partial(get_correlations, sent_vec_func=get_sentence_vec)
    output = get_summarization_simple(news, score_func_embed, 200)
    print(output)

    ## method 3, results similar to method 2
    topic_words = get_topic_words(news, stop_words, 15)
    topic_vec = get_sentence_vec(' '.join(topic_words))
    score_func_topic = partial(get_correlations, sent_vec_func=get_sentence_vec, text_vec=topic_vec)
    output = get_summarization_simple(news, score_func_topic, 200)
    print(output)
