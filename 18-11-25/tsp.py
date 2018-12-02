import math
import random
import time
from collections import defaultdict

import networkx as nx
import numpy as np

x = [random.randint(0, 100) for i in range(10)]
y = [random.randint(0, 100) for i in range(10)]


def mymemo(func):
    di = {}

    def __wraps(*args):
        my_key = str(args)
        if my_key not in di.keys():
            # print('key:', my_key)
            val = func(*args)
            di[my_key] = val
            return val
        else:
            return di[my_key]

    return __wraps


dis = np.zeros((len(x), len(x)))
for i in range(len(x)):
    for j in range(len(x)):
        dis[i][j] = math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)


def tsp_solution(d):
    @mymemo
    def tsp_solve(node, node_set):
        if node_set:
            return min([(d[n][node] + tsp_solve(n, node_set - set([n]))[0], n) for n in node_set], key=lambda x: x[0])
        else:
            return (d[0][node], 0)

    min_path = []
    min_path.append(0)
    n = 0
    ns = set(range(1, len(d)))
    while True:
        l, ln = tsp_solve(n, ns)
        if ln == 0:
            break
        min_path.append(ln)
        n = ln
        ns = ns - set([ln])

    min_path.append(0)

    return min_path


def get_path_len(path):
    dt = 0
    for i in range(1, len(path)):
        dt += dis[path[i]][path[i - 1]]
    return dt


positions = {}
for i in range(len(x)):
    positions[i] = (x[i], y[i])


def draw_path(path):
    pg = defaultdict(list)
    for i, p in enumerate(path):
        if i != len(path) - 1:
            pg[p].append(path[i + 1])

    graph = nx.Graph(pg)
    nx.draw(graph, pos=positions, with_labels=True, node_size=20, edge_color='green', width=2.0, font_size=20)


def main():
    start_time = time.time()
    tour = tsp_solution(dis)
    print('used time:', time.time() - start_time)
    print('path:', ' -> '.join(str(t) for t in tour))
    print('len:', get_path_len(tour))
    draw_path(tour)


if __name__ == '__main__':
    main()
