graph = {
        'A': 'B B B C',
        'B': 'A C',
        'C': 'A B D E',
        'D': 'C',
        'E': 'C F',
        'F': 'E'
        }

for k in graph:
    lst = []
    for i in graph[k].split():
        if i not in lst:
            lst.append(i)
    graph[k] = lst
print(graph)

seen = []
need_visit = ['A']
while need_visit:
    node = need_visit.pop(0)
    if node in seen: continue
    print('I am looking at: {}'.format(node))
    need_visit += graph[node]
    seen.append(node)

