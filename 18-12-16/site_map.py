import re
from urllib import request

import networkx as nx

pat = re.compile('href="(http://www\.cmiw\.cn/(?:forum|thread)-.*?\.html)')


def download_url(url):
    return request.urlopen(url).read().decode('gb18030')


def write_to_file(content, filename):
    with open(filename, 'a', encoding='gb18030') as fout:
        fout.write(content)


def dict2file(di, filename):
    with open(filename, 'w', encoding='utf-8') as fout:
        for d in di:
            fout.write(str(d))
            fout.write('->')
            size = len(di[d])
            for i, v in enumerate(di[d]):
                fout.write(v)
                if i != size - 1:
                    fout.write(',')
            fout.write('\n')


def get_page_links(url):
    content = download_url(url)
    refs = re.findall(pat, content)
    ret = []
    for p in refs:
        if p not in ret:
            ret.append(p)
    return ret


if __name__ == '__main__':
    site_map = {}
    seen = set()
    start = ['http://www.cmiw.cn/']

    count = 0

    while start:
        current = start.pop(0)
        if current in seen:
            continue

        print('process page\t{}, {}'.format(count, current))

        page_links = get_page_links(current)
        seen.add(current)
        site_map[current] = page_links

        start += page_links
        count += 1
        if count >= 1000:
            break

    dict2file(site_map, 'site_map.txt')

    graph = nx.Graph(site_map)
    pageranks = sorted(nx.pagerank(graph).items(), key=lambda x: x[1], reverse=True)
    with open('pageranks.txt', 'w', encoding='utf-8') as fout:
        for v in pageranks:
            fout.write('{}, {}\n'.format(v[0], v[1]))
