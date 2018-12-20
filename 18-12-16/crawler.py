import os
import re
import sys
from urllib import request

html_re = re.compile('>([\s\S]*?)<')


def download_url(url):
    return request.urlopen(url).read().decode('gb18030')


def write_to_file(content, filename):
    with open(filename, 'a', encoding='gb18030') as fout:
        fout.write(content)


def get_index_pages(ic):
    page_list = re.findall(r'(http\://www\.cmiw\.cn/forum-\d+-1\.html)', ic)
    ret = []
    for p in page_list:
        if p not in ret:
            ret.append(p)
    return ret


def get_forum_pages_from_page_content(pc):
    total_page = re.findall(r'totalpage="(\d+)"', pc)
    if total_page:
        total_page = int(total_page[0])
    else:
        total_page = 1

    page_pre = re.findall(r'(http\://www\.cmiw\.cn/forum-\d+-)1\.html', page)[0]
    forum_pages = [page_pre + str(i) + '.html' for i in range(1, total_page + 1)]
    return forum_pages


def get_thread_list_from_page_content(pc):
    thl = re.findall(r'(http\://www\.cmiw\.cn/thread-\d+-\d+-\d+\.html)', pc)
    thurls = []
    for t in thl:
        if t not in thurls:
            thurls.append(t)
    return thurls


def extract_content_from_thread(url):
    tc = download_url(url)
    cc = re.findall(r'id="postmessage_\d+"([\s\S]*?)/td></tr></table>', tc)
    ret = ''
    for c in cc:
        ci = ''.join(html_re.findall(c))
        ci = ci.replace('&nbsp;', '')
        ci = ci.replace('&amp;', '')
        ci = ci.replace('nbsp;', '')
        ci = ci.replace('\n\n', '\n')
        ret = ret + ci.strip() + '\n'
    return ret


if __name__ == '__main__':
    content = download_url('http://www.cmiw.cn/')
    page_urls = get_index_pages(content)
    count = 0
    for i, page in enumerate(page_urls):
        if i > 0:
            break
        page_content = download_url(page)
        forum_pages = get_forum_pages_from_page_content(page_content)
        for fp in forum_pages:
            fpc = download_url(fp)
            thread_urls = get_thread_list_from_page_content(fpc)
            for url in thread_urls:
                tc = extract_content_from_thread(url)
                tcl = tc.split('\n')
                fn = re.findall(r'(thread.+?\.html)', url)
                if fn:
                    if not os.path.exists(fn[0]):
                        for el in tcl:
                            if el.strip():
                                write_to_file(el.strip() + '\n', fn[0])
                        count += 1
                        print('process file:', count)
                        if count > 10000:
                            sys.exit(0)
