import urllib2
import re
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
def tran(word):
    url='http://dict.baidu.com/s?wd={0}&tn=dict'.format(word)
    print url
    req=urllib2.Request(url)
    resp=urllib2.urlopen(req)
    resphtml=resp.read()
    text = re.search(r'explain: "(.*)"',resphtml)
    return text.group(1).replace('<br />',' ')
a=tran('哈哈')

