import re
import requests
from collections import defaultdict

# import networkx as nx

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

        filename = 'path_' + str(i + 1) + '.txt'
        # with open(filename, 'w', encoding='utf-8') as f:
        #     f.write(need_content[0])
        # print('=====================')
        # print('path: ', path_list[i][1])
        path = re.findall(path_patterns[i], need_content[0])
        with open('paths.txt', 'a', encoding='utf-8') as f:
            # f.write('=====================\n')
            f.write('path:{}\n'.format(path_list[i][1]))
            for (start, to, length) in path:
                if float(length) < 10: length = str(int(float(length) * 1000))  # 网页内容中有3千米,1.6这样的字样
                # print('from:{}, to:{}, len:{}'.format(start, to, length))
                length_dict[(start, to)] = int(length)
                if to not in connection_dict[start]:
                    connection_dict[start].append(to)
                if start not in connection_dict[to]:
                    connection_dict[to].append(start)
                f.write('{}->{}->{}\n'.format(start, to, length))

with open('connection.txt', 'w', encoding='utf-8') as f:
    for d in connection_dict.keys():
        f.write('{} -> \t'.format(d))
        for v in connection_dict[d]:
            f.write(v + '\t')
        f.write('\n')
