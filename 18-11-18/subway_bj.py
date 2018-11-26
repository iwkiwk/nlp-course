import re
import requests
from collections import defaultdict
import networkx as nx

headers = {"User-Agent": "User-Agent:Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;"}
url_subway = 'https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485'

content = requests.get(url_subway, headers=headers).content.decode('utf8').replace('\n', '')
# print(content[:200])

## Get stations' name and url ##
str_beg = '<h3 class="title-text"><span class="title-prefix">北京地铁</span>运行时间</h3>'
str_end = '参考资料：<sup class="sup--normal" data-sup="55">'
pattern = re.compile(str_beg + '(.*)' + str_end)

need_content = re.findall(pattern, content)[0]
# print(need_content)

subway_pat = re.compile('href="(.+?)">(.+?)</td>')
subway_paths = re.findall(subway_pat, need_content)
path_list = []

for (url, name) in subway_paths:
    # print(url)
    # print(name)
    url = 'https://baike.baidu.com' + url
    name = name.replace('</a>', '')
    path_list.append((url, name))

for (url, name) in path_list:
    print('url: ', url)
    print('name: ', name)

path_num = len(path_list)

str_begs = ['起始/终到车站'] * 18
str_begs[11] = '14号线（东段）相邻站间距信息统计表'  # 14号线分东西两段
str_begs[13] = '站区间<'

str_ends = ['相邻站间距信息'] * 18
str_ends[2] = '站间公里数资料来源'
str_ends[10] = '相邻站间距信息统计表'
str_ends[13] = 'name="首末车时间"'

path_patterns = ['([\u4E00-\u9FA5]+)—*[—~～]([\u4E00-\u9FA5]+).*?>.*?>(\d+\.?\d*)'] * path_num

length_dict = {}
connection_dict = defaultdict(list)

## Get stations' information ##
for i in range(len(path_list)):
    content = requests.get(path_list[i][0], headers=headers).content.decode('utf8').replace('\n', '')
    str_beg = str_begs[i]
    str_end = str_ends[i]
    pattern = re.compile(str_beg + '(.*)' + str_end)
    need_content = re.findall(pattern, content)

    if need_content:

        filename = 'station_' + str(i + 1) + '.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(need_content[0])
        print('=====================')
        print('station: ', path_list[i][1])
        path = re.findall(path_patterns[i], need_content[0])
        # with open('paths.txt', 'a', encoding='utf-8') as f:
        #     f.write('=====================\n')
        #     f.write('station: {}\n'.format(path_list[i][1]))
        for (start, to, length) in path:
            if float(length) < 10: length = str(int(float(length) * 1000))  # 网页内容中有3千米,1.6这样的字样
            # print('from:{}, to:{}, len:{}'.format(start, to, length))
            length_dict[(start, to)] = int(length)
            if to not in connection_dict[start]:
                connection_dict[start].append(to)
            if start not in connection_dict[to]:
                connection_dict[to].append(start)
            # f.write('from:{}, to:{}, len:{}\n'.format(start, to, length))

subway_graph = nx.Graph(connection_dict)
nx.draw(subway_graph, with_labels=True, node_size=10)


def get_station_dis(station1, station2):
    if (station1, station2) in length_dict.keys():
        return length_dict[(station1, station2)]
    elif (station2, station1) in length_dict.keys():
        return length_dict[(station2, station1)]
    else:
        return float('inf')


get_station_dis('苹果园', '古城')


def search_graph(graph, start, destination):
    paths = [[start]]
    seen = set()
    while paths:
        path = paths.pop(0)
        frontier = path[-1]
        for station in graph[frontier]:
            if station in seen:
                continue

            new_path = path + [station]
            if station == destination:
                return new_path
            paths.append(new_path)
        seen.add(frontier)
    return []


def is_goal(node, dest):
    return node == dest


def get_successor(graph, node):
    return graph[node]


def get_path_distance(path):
    dis = 0
    for i in range(1, len(path)):
        dis += get_station_dis(path[i], path[i - 1])
    return dis


way = search_graph(connection_dict, '青年路', '七里庄')
print(way)
get_path_distance(way)


## Dijkstra ##
def dijkstra(graph, source, dest):
    results = []
    included = []
    for vertex in graph.keys():
        lt = []
        lt.append(vertex)

        if vertex == source:
            lt.append(0)
            lt.append(None)
            included.append(True)
        elif vertex in graph[source]:
            lt.append(get_station_dis(source, vertex))
            lt.append(source)
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
    path = get_final_path(ans, source, dest)

    return path


path = dijkstra(connection_dict, '青年路', '七里庄')
print(list(reversed(path)))
get_path_distance(path)

# with open('len.txt', 'w', encoding='utf-8') as f:
#     for d in length_dict.keys():
#         f.write('{}\t{}\t{}\n'.format(d[0], d[1], str(length_dict[d])))

# print(connection_dict)
#
# with open('connection.txt', 'w', encoding='utf-8') as f:
#     for d in connection_dict.keys():
#         f.write('{} -> \t'.format(d))
#         for v in connection_dict[d]:
#             f.write(v + '\t')
#         f.write('\n')

# len(subway_graph)

# with open('subway_lat_long.txt', 'r', encoding='utf-8') as f:
#     subway_geo_content = f.read()
#
# subway_lon_lat = {}
# subway_geo_list = subway_geo_content.split('|')
# for si in subway_geo_list:
#     t = si.split(',')
#     subway_lon_lat[t[0]] = (float(t[1]), float(t[2]))
#
# print(subway_lon_lat)
#
# len(subway_lon_lat)
#
# connection_dict_trim = defaultdict(list)
# for k in subway_lon_lat.keys():
#     if k in connection_dict.keys():
#         l = connection_dict[k]
#         for ll in l:
#             if ll in subway_lon_lat.keys():
#                 if ll not in connection_dict_trim[k]:
#                     connection_dict_trim[k].append(ll)
#                 if k not in connection_dict_trim[ll]:
#                     connection_dict_trim[ll].append(k)
#
# print(connection_dict_trim)
#
# len(connection_dict_trim)
# graph_t = nx.Graph(connection_dict_trim)
# nx.draw(graph_t, pos=subway_lon_lat, with_labels=True, node_size=10)
