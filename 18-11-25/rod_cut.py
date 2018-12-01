import time
from collections import defaultdict
from functools import lru_cache

from memo import memo

prices = defaultdict(lambda: -float('inf'))
for i, v in enumerate([1, 5, 8, 9, 10, 17, 17, 20, 24, 30]):
    prices[i + 1] = v
print(prices)


def mymemo(func):
    di = {}

    def __wraps(*args):
        my_key = args[0]
        if my_key not in di.keys():
            val = func(*args)
            di[my_key] = val
            return val
        else:
            return di[my_key]

    return __wraps


cut_solution = {}
cut_solution_memo = {}
cut_solution_lru = {}


@mymemo
def reveneu(n):
    split, r = max([(0, prices[n])] + [(i, reveneu(n - i) + reveneu(i)) for i in range(1, n)], key=lambda x: x[1])
    cut_solution[n] = (split, n - split)
    return r


@memo
def reveneu_memo(n):
    split, r = max([(0, prices[n])] + [(i, reveneu_memo(n - i) + reveneu_memo(i)) for i in range(1, n)],
                   key=lambda x: x[1])
    cut_solution_memo[n] = (split, n - split)
    return r


@lru_cache(1000)
def reveneu_lru(n):
    split, r = max([(0, prices[n])] + [(i, reveneu_lru(n - i) + reveneu_lru(i)) for i in range(1, n)],
                   key=lambda x: x[1])
    cut_solution_lru[n] = (split, n - split)
    return r


def parse_solution(i, solution):
    left, right = solution[i]
    if left == 0: return [right]
    return [left] + parse_solution(right, solution)


start_time = time.time()
print(reveneu(100))
print('mymemo run time:{}'.format(time.time() - start_time))
print(cut_solution)
print(parse_solution(85, cut_solution))

start_time = time.time()
print(reveneu_memo(100))
print('memo run time:{}'.format(time.time() - start_time))
print(cut_solution_memo)
print(parse_solution(85, cut_solution_memo))

start_time = time.time()
print(reveneu_lru(100))
print('lru_cache run time:{}'.format(time.time() - start_time))
print(cut_solution_lru)
print(parse_solution(85, cut_solution_lru))
