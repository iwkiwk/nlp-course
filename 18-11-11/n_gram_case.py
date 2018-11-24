# Preprocess:
#
# 1. Extract wiki data
#    $ python WikiExtractor.py -b 1024M -o extracted zhwiki-20181101-pages-articles.xml.bz2
#
# 2. Covert traditional chinese to simplified chinese
#    Using opencc
#    $ opencc -i wiki_00 -o zh_wiki_00 -c zht2zhs.ini
#    $ opencc -i wiki_01 -o zh_wiki_01 -c zht2zhs.ini
#
# 3. Replace special characters
#    import re
#    import sys
#    import codecs
#    def myfun(input_file):
#        p1 = re.compile(r'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-')
#        p2 = re.compile(r'[（\(][，；。？！\s]*[）\)]')
#        p3 = re.compile(r'[「『]')
#        p4 = re.compile(r'[」』]')
#        outfile = codecs.open('std_' + input_file, 'w', 'utf-8')
#        with codecs.open(input_file, 'r', 'utf-8') as myfile:
#            for line in myfile:
#                line = p1.sub(r'\2', line)
#                line = p2.sub(r'', line)
#                line = p3.sub(r'“', line)
#                line = p4.sub(r'”', line)
#                outfile.write(line)
#        outfile.close()
#    if __name__ == '__main__':
#        if len(sys.argv) != 2:
#            print("Usage: python script.py inputfile")
#            sys.exit()
#        input_file = sys.argv[1]
#        myfun(input_file)
#
# 4. Remove special lines start with '<', i.e. '<doc id****', '<\doc>'
#    import re
#    import codecs
#    import sys
#
#    r1 = re.compile(r'^<')
#
#    def pre(input_file):
#        outfile = codecs.open(input_file + '_pre', 'w', 'utf-8')
#        with codecs.open(input_file, 'r', 'utf-8') as myfile:
#            for line in myfile:
#                if re.match(r1, line):
#                    pass
#                else:
#                    outfile.write(line)
#        outfile.close()
#
#    if __name__ == '__main__':
#        if len(sys.argv) != 2:
#            sys.exit()
#
#        pre(sys.argv[1])

from collections import Counter
from matplotlib.pyplot import xscale, yscale, plot, title
from functools import reduce
import re
import time
import copy
import jieba


def tokenize(string):
    return ''.join(re.findall('[\w|\d]+', string))


def get_prob_from_counter(counter):
    all_occurs = sum(counter.values())

    def get_item_prob(item):
        return counter[item] / all_occurs

    return get_item_prob


def get_running_time(func, args, times):
    start_time = time.time()
    for i in range(times):
        func(args)
    print('{} elapsed time: {}'.format(func.__name__, (time.time() - start_time) / times))


def mult_func(a, b):
    return a * b


def get_string_probablity(func, string):
    return reduce(mult_func, [func(s) for s in string])


def get_probability_performance(prob_func, func, pairs):
    for (p1, p2) in pairs:
        print('{} with probablity: {}'.format(p1, prob_func(func, tokenize(p1))))
        print('{} with probablity: {}'.format(p2, prob_func(func, tokenize(p2))))
        print('')


# N-gram counter
def gen_n_gram_counter(string, gram_n=2):
    return Counter([''.join(string[i:i + gram_n]) for i in range(len(string) - gram_n + 1)])


# Good–Turing frequency estimation
def gen_gd_counter(counter):
    counter_counter = Counter([n for (w, n) in counter.items()])
    # deep copy
    new_counter = copy.deepcopy(counter)
    for (w, n) in counter.items():
        if n < 8:
            new_counter[w] = (n + 1) * counter_counter[n + 1] / counter_counter[n]
    return new_counter


def get_two_gram_gd_prob(counter1, counter2, pairs, content):
    counter1_gd = gen_gd_counter(counter1)
    counter2_gd = gen_gd_counter(counter2)

    get_uni_prob = get_prob_from_counter(counter1_gd)
    get_pair_prob = get_prob_from_counter(counter2_gd)

    # def get_2_gram_prob(word, prev):
    #     if get_pair_prob(prev + word) > 0:
    #         return get_pair_prob(prev + word) / get_uni_prob(prev)
    #     else:
    #         return get_uni_prob(word)
    #
    # def get_2_gram_string_prob(string):
    #     probablities = []
    #     for i, c in enumerate(string):
    #         prev = '<s>' if i == 0 else string[i - 1]
    #         probablities.append(get_2_gram_prob(c, prev))
    #     return reduce(mult_func, probablities)
    #
    # for (p1, p2) in pairs:
    #     print('{} with probablity: {}'.format(p1, get_2_gram_string_prob(tokenize(p1))))
    #     print('{} with probablity: {}'.format(p2, get_2_gram_string_prob(tokenize(p2))))
    #     print('')

    N_total = sum(counter2.values())
    N_seen = sum(counter2_gd.values())

    def get_prob(validate_str):
        probs = []
        unseens = []
        for i in range(1, len(validate_str)):
            cur = validate_str[i - 1:i + 1]
            if cur not in content:
                unseens.append(cur)
            else:
                prob = get_pair_prob(cur) / get_uni_prob(cur[0])
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


pair1 = """前天晚上吃晚饭的时候
前天晚上吃早饭的时候""".split('\n')

pair2 = """正是一个好看的小猫
真是一个好看的小猫""".split('\n')

pair3 = """我无言以对，简直
我简直无言以对东磁哇啦嚄""".split('\n')

pairs = [pair1, pair2, pair3]

filename = 'std_zh_wiki_01_pre'
with open(filename, encoding='utf-8') as f:
    all_content = f.read()

all_content = tokenize(all_content)

#### N-gram using chars ####

# Unigram
uni_gram_counter = Counter(all_content)
print(uni_gram_counter.most_common(10))

M = uni_gram_counter.most_common()[0][1]
len(uni_gram_counter)

xscale('log')
yscale('log')

plot([c for (w, c) in uni_gram_counter.most_common()])
plot([M / i for i in range(1, len(uni_gram_counter) + 1)])

get_item_prob = get_prob_from_counter(uni_gram_counter)
get_item_prob('的')

get_running_time(get_item_prob, '的', 100000)

get_string_probablity(get_item_prob, '你好吗')
get_string_probablity(get_item_prob, '你坏吗')

get_probability_performance(get_string_probablity, get_item_prob, pairs)

# 2-gram
two_gram_counter = gen_n_gram_counter(all_content)
print(two_gram_counter.most_common(10))
get_two_gram_gd_prob(uni_gram_counter, two_gram_counter, pairs, all_content)

#### N-gram using words ####

seg_list = jieba.lcut(all_content, cut_all=False)
print(seg_list[0:100])

words_counter = Counter(seg_list)
words_counter.most_common(1000)

all_words_occurs = sum(words_counter.values())

get_word_prob = get_prob_from_counter(words_counter)

words_counter['的']
get_word_prob('的')
get_word_prob('东磁')

print(pairs)
for p in pairs:
    l1 = jieba.lcut(tokenize(p[0]), cut_all=False)
    l2 = jieba.lcut(tokenize(p[1]), cut_all=False)
    p1 = 1
    p2 = 1
    for i1 in l1:
        p1 *= get_word_prob(i1)

    for i2 in l2:
        p2 *= get_word_prob(i2)

    print('{} with probablity: {}'.format(p[0], p1))
    print('{} with probablity: {}'.format(p[1], p2))
    print('')

list2 = []
for i in range(1, len(seg_list)):
    list2.append(seg_list[i - 1] + seg_list[i])

words_counter2 = Counter(list2)
words_counter2.most_common(100)

get_word_prob2 = get_prob_from_counter(words_counter2)

for p in pairs:
    l1 = jieba.lcut(tokenize(p[0]), cut_all=False)
    l2 = jieba.lcut(tokenize(p[1]), cut_all=False)

    ll1 = []
    ll2 = []
    for i1 in range(1, len(l1)):
        ll1.append(l1[i1 - 1] + l1[i1])
    for i2 in range(1, len(l2)):
        ll2.append(l2[i2 - 1] + l2[i2])

    # print('ll1: ', ll1)
    # print('ll2: ', ll2)

    p1 = 1
    p2 = 1
    for i1 in ll1:
        p1 *= get_word_prob(i1)

    for i2 in ll2:
        p2 *= get_word_prob(i2)

    print('{} with probablity: {}'.format(p[0], p1))
    print('{} with probablity: {}'.format(p[1], p2))
    print('')


def get_two_gram_gd_prob_for_words(counter1, counter2, pairs, content):
    counter1_gd = gen_gd_counter(counter1)
    counter2_gd = gen_gd_counter(counter2)

    get_uni_prob = get_prob_from_counter(counter1_gd)
    get_pair_prob = get_prob_from_counter(counter2_gd)

    N_total = sum(counter2.values())
    N_seen = sum(counter2_gd.values())

    def get_prob(validate_str):
        probs = []
        unseens = []

        valis = jieba.lcut(validate_str, cut_all=False)
        for i in range(1, len(valis)):
            cur = valis[i - 1] + valis[i]
            if cur not in content:
                unseens.append(cur)
            else:
                prob = get_pair_prob(cur) / get_uni_prob(cur[0])
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


get_two_gram_gd_prob_for_words(words_counter, words_counter2, pairs, seg_list)
