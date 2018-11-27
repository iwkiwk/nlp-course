from collections import defaultdict
import re
import networkx as nx
import matplotlib.pyplot as plt
import math

subway_paths = defaultdict(list) # 线路及包含站点
subway_connections = defaultdict(list) # 站点连接
with open('paths.txt', 'r', encoding='utf-8') as f:
    path_name = ''
    for line in f:
        if line.startswith('path'):
            path_name = line.split(':')[1].strip()
        else:
            connect = line.split('->')
            start = connect[0]
            to = connect[1]
            if start not in subway_paths[path_name]:
                subway_paths[path_name].append(start)
            if to not in subway_paths[path_name]:
                subway_paths[path_name].append(to)

            if to not in subway_connections[start]:
                subway_connections[start].append(to)
            if start not in subway_connections[to]:
                subway_connections[to].append(start)

station_path = defaultdict(list) # 站点所在线路
for name in subway_paths.keys():
    for node in subway_paths[name]:
        if name not in station_path[node]:
            station_path[node].append(name)

# print(station_path)
# print(subway_paths)
# print(subway_connections)

# station_names = []
# for k in subway_paths.keys():
#     for name in subway_paths[k]:
#         if name not in station_names:
#             station_names.append(name)
# with open('stations.txt', 'w', encoding='utf-8') as f:
#     for name in station_names:
#         f.write('北京' + name + '地铁站\n')

# 根据经纬度计算大圆距离
def geo_distance(origin, destination):
    """
    Calculate the Haversine distance.
    """
    lon1, lat1 = origin
    lon2, lat2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


stations_lon_lat = {}
with open('station_latlon.txt', 'r', encoding='utf-8') as f:
    for line in f:
        l = line.split(',')
        name = l[0]
        lat = float(l[1])  # latitude
        lon = float(l[2])  # longitude
        stations_lon_lat[name] = (lon, lat)

subway_graph = nx.Graph(subway_connections)
nx.draw(subway_graph, pos=stations_lon_lat, with_labels=True, node_size=10)
# nx.draw(subway_graph, pos=stations_lon_lat, node_size=10)

# 获得站点距离
def get_station_dis(station1, station2):
    return geo_distance(stations_lon_lat[station1], stations_lon_lat[station2])

# 获得路线长度
def get_path_length(path):
    dis = 0
    for i in range(1, len(path)):
        dis += get_station_dis(path[i], path[i - 1])
    return dis

# 估计总长度
def estimate_dis(path, dest):
    return get_station_dis(path[-1], dest) + get_path_length(path)


def search_path(graph, start, dest, stragety):
    paths = [[start]]
    seen = set()
    chosen_path = []
    while paths:
        path = paths.pop(0)
        frontier = path[-1]
        if frontier in seen:
            continue

        for node in graph[frontier]:
            if node in path:
                continue

            new_path = path + [node]
            paths.append(new_path)
            if node == dest:
                # chosen_path.append(new_path)
                return new_path
        paths = stragety(paths, dest)
        # print([len(p) for p in paths])
        # print([estimate_dis(p, dest) for p in paths])
        seen.add(frontier)
    return chosen_path

# 返回某条线上的两点路径
def get_path_on_one_way(pn, src, dest):
    i = subway_paths[pn].index(src)
    j = subway_paths[pn].index(dest)
    if i < j:
        return subway_paths[pn][i:j+1]
    else:
        return list(reversed(subway_paths[pn][j:i+1]))

# 搜索最少走哪几条线
def search_min_transfer(station_graph, path_graph, start, dest):
    start_paths = station_path[start] # 站点可能在多条线路上
    dest_paths = station_path[dest]
    path_nums = []
    for i in range(len(start_paths)):
        for j in range(len(dest_paths)):
            num = len(search_path(path_graph, start_paths[i], dest_paths[j], min_transfers))
            path_nums.append(num)
    ind = path_nums.index(min(path_nums))
    i = ind // len(dest_paths)
    j = ind % len(dest_paths)
    # 走哪几条线换乘最少
    path_path = search_path(path_graph, start_paths[i], dest_paths[j], min_transfers)
    print('Minimum transfer:', path_path)
    crosses = []
    for i in range(1, len(path_path)):
        crosses.append(get_path_intersection(path_path[i], path_path[i-1])[0])
    pass_stations = []
    pass_stations.append(start)
    pass_stations += crosses
    pass_stations.append(dest)
    # print(pass_stations)
    paths = []
    for i in range(len(pass_stations)-1):
        path = get_path_on_one_way(path_path[i], pass_stations[i], pass_stations[i+1])
        paths += path[:-1]
    paths.append(dest)
    return paths


def sort_paths(paths, func, beam):
    return sorted(paths, key=func)[:beam]

# 最短路径策略A*
def min_distance(paths, dest):
    return sort_paths(paths, lambda p: get_path_length(p) + get_station_dis(p[-1], dest), 20)

# 最少节点，对于线路连接是换乘少，对于站点连接是站点少
def min_transfers(paths, dest):
    return sort_paths(paths, lambda p: len(p), 10)

def max_transfers(paths, dest):
    return sort_paths(paths, lambda p: -len(p), 10)


# 获得线路交点
def get_path_intersection(p1, p2):
    cross = []
    for p in subway_paths[p1]:
        if p in subway_paths[p2]:
            cross.append(p)
    return cross

# 判断是否有交点
def is_paths_crossed(p1, p2):
    for p in subway_paths[p1]:
        if p in subway_paths[p2]:
            return True
    return False

# 获取线路连接
def get_path_connections(paths):
    connect = defaultdict(list)
    for p1 in paths.keys():
        for p2 in paths.keys():
            if p1 == p2: continue
            is_crossed = is_paths_crossed(p1, p2)
            if is_crossed:
                if p2 not in connect[p1]:
                    connect[p1].append(p2)
                if p1 not in connect[p2]:
                    connect[p2].append(p1)
    return connect

## Dijkstra ##
## for comparsion with A* ##
def dijkstra(graph, start, dest):
    results = []
    included = []
    for vertex in graph.keys():
        lt = []
        lt.append(vertex)

        if vertex == start:
            lt.append(0)
            lt.append(None)
            included.append(True)
        elif vertex in graph[start]:
            lt.append(get_station_dis(start, vertex))
            lt.append(start)
            included.append(False)
        else:
            lt.append(float('inf'))
            lt.append(None)
            included.append(False)

        results.append(lt)

    # print(results)
    # print(included)

    while False in included:
        indices = []
        for i in range(len(included)):
            if not included[i]:
                indices.append(i)

        dis = [results[i][1] for i in indices]
        min_dis_ind = indices[dis.index(min(dis))]
        included[min_dis_ind] = True
        F = results[min_dis_ind][0]

        for i in range(len(included)):
            if not included[i]:
                T = results[i][0]
                if T in graph[F]:
                    new_dis = results[min_dis_ind][1] + get_station_dis(F, T)
                    if new_dis < results[i][1]:
                        results[i][1] = new_dis
                        results[i][2] = F

    # print(results)
    # return results

    def gen_dict_from_results(result):
        ret = {}
        for r in result:
            ret[r[0]] = r
        return ret

    def get_final_path(result, s, d):
        path = []
        path.append(d)
        node = result[d][2]
        while node != s:
            path.append(node)
            node = result[node][2]
        path.append(s)
        return path

    ans = gen_dict_from_results(results)
    path = get_final_path(ans, start, dest)

    return list(reversed(path))


get_station_dis('八宝山', '五棵松')
get_path_length(['八宝山', '玉泉路', '五棵松'])

estimate_dis(['五棵松', '玉泉路'], '青年路')
estimate_dis(['五棵松', '万寿路'], '青年路')

min_distance([['五棵松', '玉泉路'], ['五棵松', '万寿路']], '青年路')

result1 = search_path(subway_connections, '青年路', '七里庄', min_distance)
result2 = dijkstra(subway_connections, '青年路', '七里庄')
result3 = search_path(subway_connections, '石门', '西单', min_distance)
result4 = dijkstra(subway_connections, '石门', '西单')
result5 = search_path(subway_connections, '昌平', '分钟寺', min_distance)
result6 = dijkstra(subway_connections, '昌平', '分钟寺')

result7 = search_path(subway_connections, '青年路', '七里庄', min_transfers)
result8 = search_path(subway_connections, '石门', '西单', min_transfers)
result9 = search_path(subway_connections, '昌平', '分钟寺', min_transfers)

print(result1)
print(result2)
print(result3)
print(result4)
print(result5)
print(result6)
print(result7)
print(result8)
print(result9)

print(get_path_length(result1))
print(get_path_length(result2))
print(get_path_length(result3))
print(get_path_length(result4))
print(get_path_length(result5))
print(get_path_length(result6))
print(len(result7))
print(len(result8))
print(len(result9))

print(get_path_intersection('北京地铁6号线', '北京地铁2号线'))
print(get_path_intersection('北京地铁1号线', '北京地铁八通线'))
print(is_paths_crossed('北京地铁1号线', '北京地铁八通线'))

subway_path_connections = get_path_connections(subway_paths)
print(subway_path_connections)

# path_connections_graph = nx.Graph(subway_path_connections)
# nx.draw(path_connections_graph, with_labels=True, node_size=10)

result10 = search_path(subway_path_connections, '北京地铁1号线', '北京地铁昌平线', min_transfers)
result11 = search_path(subway_path_connections, '北京地铁1号线', '北京地铁昌平线', max_transfers)
print(result10)
print(result11)

result12 = search_min_transfer(subway_connections, subway_path_connections, '昌平', '分钟寺')
print(result12)

