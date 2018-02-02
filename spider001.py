#-*- coding: utf-8 -*-
import re
import os
import requests
from urllib import quote, unquote


def download_pic(html, keyword , i ):
    pic_url = re.findall('"objURL":"(.*?)",', html)
    print '找到关键词:' + unquote(keyword) + '的图片，现在开始下载图片...'
    for each in pic_url:
        print u'正在下载第' + str(i + 1) + u'张图片，图片地址:' + str(each)
        try:
            pic = requests.get(each, timeout=50)
        except Exception, ex:
            print u'【错误】当前图片无法下载'
            continue
        file_name = 'pictures/' + unquote(keyword) + '_' + str(i + 1) + '.jpg'
        fp = open(file_name, 'wb')
        fp.write(pic.content)
        fp.close()
        i += 1
    return i

if __name__ == '__main__':
    word =  raw_input('Input keywords:')
    word = quote(word)
    pnMax = input('Input max pn:')
    pncount = 0
    if not os.path.exists('pictures'):
        os.mkdir('pictures')
    while pncount < pnMax:
        str_pn = str(pncount)
        url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='+word+'&pn='+str_pn+'&ct=&ic=0&lm=-1&width=0&height=0'
        result = requests.get(url)
        pncount = download_pic(result.text, word, pncount)
    print u'下载完毕'
