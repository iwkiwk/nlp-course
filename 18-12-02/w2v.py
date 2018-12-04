import jieba
import os
import time
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence


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


if __name__ == '__main__':
    w2v_model = get_model_from_file()
    print(w2v_model.wv['数学'])
    print(w2v_model.wv.most_similar('数学'))
