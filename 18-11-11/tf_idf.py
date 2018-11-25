# Sentence probability estimating using TF-IDF strategy
# Better result on common words like '的'.

from collections import Counter
import re
import jieba
import copy
from functools import reduce
import math


def tokenize(string):
    return ''.join(re.findall('[\w|\d]+', string))


filename = 'std_zh_wiki_01'

content_list = []
with open(filename, encoding='utf-8') as f:
    while True:
        line = f.readline()
        if not line:
            break
        if line.startswith('<doc id'):
            content = ''
            while True:
                next = f.readline()
                if next.startswith('</doc>'):
                    break
                else:
                    content += tokenize(next)
            content_list.append(content)

doc_num = len(content_list)


def inv_doc_freq(item):
    n = 0
    for content in content_list:
        if item in content:
            n += 1
    n = max(1, n)
    return math.log(doc_num / n, 10)


# Good–Turing frequency estimation
def gen_gd_counter(counter):
    counter_counter = Counter([n for (w, n) in counter.items()])
    # deep copy
    new_counter = copy.deepcopy(counter)
    for (w, n) in counter.items():
        if n < 8:
            new_counter[w] = (n + 1) * counter_counter[n + 1] / counter_counter[n]
    return new_counter


with open(filename, encoding='utf-8') as f:
    all_content = f.read()
all_content = tokenize(all_content)
seg_list = jieba.lcut(all_content, cut_all=False)
words_counter = Counter(seg_list)
words_counter_gd = gen_gd_counter(words_counter)

list2 = []
for i in range(1, len(seg_list)):
    list2.append(seg_list[i - 1] + seg_list[i])
words_counter2 = Counter(list2)
words_counter2_gd = gen_gd_counter(words_counter2)

uni_count = sum(words_counter.values())
two_count = sum(words_counter2.values())

validate_str0 = '昨天的红烧鱼头很好吃'
validate_str1 = '昨天的的的的的很好吃'
test_pairs = [[validate_str0, validate_str1]]


def get_term_frq1(item):
    return words_counter_gd[item] / uni_count


def get_tf_idf1(item):
    return get_term_frq1(item) * inv_doc_freq(item)


def get_term_frq2(item):
    return words_counter2_gd[item] / two_count


def get_tf_idf2(item):
    return get_term_frq2(item) * inv_doc_freq(item)


def mult_func(a, b):
    return a * b


def get_two_gram_prob_gd_tfidf(pairs, content):
    N_total = two_count
    N_seen = sum(words_counter2_gd.values())

    def get_prob(validate_str):
        probs = []
        unseens = []

        valis = jieba.lcut(validate_str, cut_all=False)
        for i in range(1, len(valis)):
            cur = valis[i - 1] + valis[i]
            if cur not in content:
                unseens.append(cur)
            else:
                prob = get_term_frq2(cur) / get_term_frq1(cur[0])
                probs.append(prob)

        print('probs 1: ', probs)

        unseens_counter = Counter(unseens)
        N_unseen = sum(unseens_counter.values())
        unseens_prob = 1 - N_seen / N_total

        for (w, c) in unseens_counter.items():
            probs.append(c * unseens_prob / N_unseen)

        print('probs 2: ', probs)

        ret = reduce(mult_func, probs)
        return ret

    for (p1, p2) in pairs:
        print('{} with probablity: {}'.format(p1, get_prob(tokenize(p1))))
        print('{} with probablity: {}'.format(p2, get_prob(tokenize(p2))))
        print('')


get_two_gram_prob_gd_tfidf(test_pairs, all_content)
