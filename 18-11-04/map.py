import networkx as nx
from matplotlib.pyplot import show, ion

BJ = 'Beijing'
SZ = 'Shenzhen'
GZ = 'Guangzhou'
WH = 'Wuhan'
HLJ= 'Heilongjiang'
NY = 'New York City'
CM = 'Chiangmai'
SG = 'Singapore'

air_route = {
        BJ: {SZ, GZ, WH, HLJ, NY},
        SZ: {BJ, SG},
        GZ: {WH, BJ, CM, SG},
        WH: {BJ, GZ},
        HLJ:{BJ},
        NY: {BJ},
        CM: {GZ}
        }

g = nx.Graph(air_route)
nx.draw(g, with_labels=True)
ion()
show()

def search_path(graph, start, dest):
    pathes = [[start]]
    seen = set()
    choosen_path = []
    while pathes:
        path = pathes.pop(0)
        prev = path[-1]
        if prev in seen: continue

        for city in graph[prev]:
            new_path = path + [city]
            pathes.append(new_path)
            if city == dest: return new_path
        seen.add(prev)
    return choosen_path

def draw_route(path): print(' -> '.join(path))

draw_route(search_path(g, SZ, CM))

