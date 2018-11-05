import networkx as nx
from matplotlib.pyplot import plot, ion, show
from functools import partial

graph = {
        '1': '2 7',
        '2': '3',
        '3': '4',
        '4': '5',
        '5': '6 10',
        '6': '5',
        '7': '8',
        '8': '9',
        '9': '10',
        '10': '5 11',
        '11': '12',
        '12': '11'
        }

for i in graph: graph[i] = graph[i].split()
print(graph)

g = nx.DiGraph()
g.add_nodes_from(graph.keys())

for k, v in graph.items():
    g.add_edges_from([(k, t) for t in v])

# Interactive mode
ion()
nx.draw(g, with_labels=True)
show()

def search(graph_, concat_func):
    seen = []
    need_visit = ['1']

    while need_visit:
        node = need_visit.pop(0)
        if node in seen: continue
        print('Looking at: {}'.format(node))
        seen.append(node)
        new_discovered = graph_[node]
        need_visit = concat_func(new_discovered, need_visit)

def new_discovered_first(new_discovered, need_visit):
    return new_discovered + need_visit

def already_discovered_first(new_discovered, need_visit):
    return need_visit + new_discovered

dfs = partial(search, concat_func=new_discovered_first)

bfs = partial(search, concat_func=already_discovered_first)

print('dfs:')
dfs(graph)
print('bfs:')
bfs(graph)

