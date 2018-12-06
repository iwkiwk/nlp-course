import jieba
import os
import time
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# Cutting words and saving to corpus file
def write_token_to_file(infile, outfile):
    words = []
    for line in open(infile, 'r', encoding='utf-8'):
        line = line.strip()
        if line:
            w = jieba.lcut(line)
            words += w + ['\n']
    outfile.writelines(' '.join(words))


def get_target_files():
    all_files = os.listdir()
    target_files = []
    for file in all_files:
        if file.endswith('pre'):
            target_files.append(file)
    return target_files


def prepare_corpus():
    with open('train_corpus.txt', 'w', encoding='utf-8') as outfile:
        for file in target_files:
            print('processing file: {}'.format(file))
            write_token_to_file(file, outfile)
            print('elapsed time:', time.time() - start_time)
            start_time = time.time()


def train_w2v_model():
    start_time = time.time()
    w2v_model = Word2Vec(LineSentence('train_corpus.txt'), workers=4)  # Using 4 threads
    w2v_model.wv.save('w2v_gensim')
    print('elapsed time:', time.time() - start_time)


def get_model_from_file():
    model = KeyedVectors.load('w2v_gensim', mmap='r')
    return model


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    ii = 0
    for word in model.wv.vocab:
        if ii > 200: break
        tokens.append(model[word])
        labels.append(word)
        ii += 1

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


if __name__ == '__main__':
    w2v_model = get_model_from_file()
    print(w2v_model.wv['数学'])
    print(w2v_model.wv.most_similar('数学'))

    font = {'family': 'simhei',
            'weight': 'regular',
            'size': '12'}
    plt.rc('font', **font)
    plt.rc('axes', unicode_minus=False)

    tsne_plot(w2v_model)
