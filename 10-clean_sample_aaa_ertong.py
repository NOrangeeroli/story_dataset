import pandas as pd
import json
import glob,os,sys
from utils import *
SENLEN = 5
'''LQZEICH='LQCYM'
RQZEICH='RQCYM'
SPLITZEICH='SCHWEIGSSS'
SPLITMARK='？ ? . 。 , ！ ! …… ，'.split(' ')
ENDMARK='？ ? 。 ！ ! ……'.split(' ')
LEGALMARK= '？ ? 。 ！ ! …… ， ,'.split(' ')
SENLEN = 5'''
'''def is_quote_complete(string):
    return string.count('“')==string.count('”')
def if_notrepeat(string):
    slotlen = min(20,int(len(string)/3.0))
    if string.find(string[:slotlen],slotlen)!= -1:
        return False
    else:
        return True
def is_improper_marks(string):
    if string.find('。：') !=-1:
        return True
    if string.find('。。。。。。') !=-1:
        return True
    return False
def is_annotation(string):
    import re
    if len(re.findall('\[.\]',string))>0:
        return True
    return False
def is_chinese(char):
    if '\u4e00' <= char <= '\u9fff':
        return True
    return False
def chinese_ratio(string):
    return len([1 for x in string if is_chinese(x)])/float(len(string))
        
def clean_spam(string):
    s = split_by(string,[' ','。','.','\xa0','_','xx','|','/'])
    return ''.join([x for x in s if len(x)>3 and chinese_ratio(x)>0.5])
def isnot_story(string):
    if string.find('读后感')!=-1:
        return True
    if string.find('读书笔记')!=-1:
        return True
    return False
def paragraph_sanity_check(string):
    return is_quote_complete(string) \
    and if_notrepeat(string) \
    and chinese_ratio(string)>0.5 \
    and not is_annotation(string)\
    and not is_improper_marks(string)\
    and not isnot_story(string)
def sentence_sanity_check(string):
    for x in string:
        if not is_chinese(x):
            if x not in LEGALMARK:
                return False
    return True
def nested_dict():
    import collections
    return collections.defaultdict(nested_dict)
def complete_quote(string):
    if string.startswith('“') and not string.endswith('”'):
        return string + '”'
    elif not string.startswith('“') and string.endswith('”'):
        return '“'+string
    else:
        return string
def is_pureword(string):
    for mark in SPLITMARK:
        if mark in string:
            return False
    return True


def is_dialogquote(string):
    if string.startswith('“') and string.endswith('”') and not is_pureword(string):
        return True
    else:
        return False
def is_normalquote(string):
    if string.startswith('“') and string.endswith('”') and is_pureword(string):
        return True
    else:
        return False
def is_sentence_complete(string):
    for mark in ENDMARK:
        if string.endswith(mark):
            return True
    return False
def split_by(string,marks):
    s = string
    for mark in marks:

        s=s.replace(mark,mark+SPLITZEICH)
    return s.split(SPLITZEICH)
def split_by_fullstop(string):
    
    return split_by(string,ENDMARK)
def split_by_quote(string):
    s=string.replace('“',SPLITZEICH+'“')
    s=s.replace('”','”'+SPLITZEICH)
    s = s.split(SPLITZEICH)
    return s
def replace_quote(string):
    s = string.replace('“',LQZEICH)
    s = s.replace('”',RQZEICH)
    return s
def undoreplace_quote(string):
    s = string.replace(LQZEICH,'“')
    s = s.replace(RQZEICH,'”')
    return s
def split_by_dialogquote(string):
    s = split_by_quote(string)
    
    s = [replace_quote(x) if is_normalquote(x) else x for x in s]
    s= ''.join(s)
    s = split_by_quote(s)
    s = [undoreplace_quote(x) for x in s]
    return s

def strip_incomplete_sentence(string):
    s = string
    

    if not is_sentence_complete(string):
        s = split_by_fullstop(string)
        s =''.join(s[:-1])
    while len(s)>1:
        # import pdb;pdb.set_trace()

        if (not is_chinese(s[0])) :
            s = s[1:]
        else:
            return s
    if len(s)==1 and not is_chinese(s):
        return ''
    return s
     
def count_clauses(string):
    return len(split_by_fullstop(string))   
def count_sentences(string):
    return len(split_by(string,ENDMARK)) '''
def split_paragraphs(string):
    s = split_by_dialogquote(string)
    s = [complete_quote(x) for x in s if len(x)>0]
    
    
    
    s = [x for x in s if not is_dialogquote(x)]
    s = [strip_incomplete_sentence(x)  for x in s if len(x)>0]
    s = [x for x in s if paragraph_sanity_check(x)]

    return s
def split_paragraphs_nur(string):
    s = split_paragraphs(string)
    # if string.startswith('爷爷年纪很大了，腿脚已不能行走，眼睛看不清了，'):
    #     import pdb;pdb.set_trace()
    ret =  [strip_incomplete_sentence(x) for x in s if len(x)>0 ]#and paragraph_sanity_check(x)]
    # ret =  [modify_illegal_sentence(x) for x in ret  ]
    # ret = del_repeats_in_list([x for x in ret ])
    # ret = [x for x in ret if len(x)<500]
    # k = lambda x : abs(len(x)-250)
    return [ret[0] ] if len(ret)>0 and len(ret[0])<300 and len(ret[0])>50 else []
def split_sentences(string):
    paralen= count_sentences(string)
    splitnum= int(paralen/SENLEN)-1
    if splitnum <= 0:
        ret =  [string,]
    else:
        s = split_by_fullstop(string)
        ret = []
        for i in range(splitnum):
            ret.append(''.join(s[i*5:i*5+5]))
        ret.append(''.join(s[splitnum*5:]))
    return [strip_incomplete_sentence(x) for x in ret if len(x)>0 and sentence_sanity_check(x)]
data_path=os.path.abspath('../../liuziqi/chinesestory/data')
# print(data_path)
pool1=glob.glob(data_path+'/aaa_ertong/*')
# print(pool1)
sentences_count=nested_dict()
backup = {}
# sentences_split=nested_dict()
paragraphs_split = nested_dict()
df = pd.DataFrame()
for fn in pool1:
    with open(fn,'r') as f:
        data=json.load(f)
    backup[data['title']] = data
    # if data['title']== '名人名言大全':
    #     import pdb;pdb.set_trace()
    samples = [(split_paragraphs_nur(c['text']),c['text']) for c in data['chapter'] if len(c['text'])<2000]
    raw_paras = sum([[x[1]]*len(x[0]) for x in samples],[])
    paras = sum([x[0] for x in samples],[])
    temp = pd.DataFrame()
    temp['paras']= pd.Series(paras)
    temp['raw_paras']= pd.Series(raw_paras)
    temp['raw_paras']= temp.apply(lambda x: x.raw_paras.replace(x.paras,'【'+x.paras+'】'),axis=1)
    temp['title'] = data['title']
    paragraphs_split[data['title']]=paras
    df = df.append(temp)
sentences_split= {k:sum([split_sentences(p) for p in paragraphs_split[k]],[]) for k in paragraphs_split.keys()}   
sentences_split=paragraphs_split
with open('aaa_ertong.json','w') as f:
    json.dump(sentences_split, f,ensure_ascii=False,indent=4)

sentence_length= [len(x) for y in sentences_split.values() for x in y]    

# print(sum([sum(x) for x in sentences_count.values()]))
import pdb;pdb.set_trace()
# sum([len(sentences_split[x]) for x in sentences_split.keys()])
    
lens = [len(x) for y in sentences_split.values() for x in y]
num = sum([len(x) for x in sentences_split.values()])
df =df.reset_index(drop= True)
for i in range(11):
    df.iloc[i*2000:(i+1)*2000].to_excel('data/data{}.xlsx'.format(i),encoding="utf_8_sig")
# df.reset_index(drop= True).to_csv('data.csv',encoding="utf_8_sig")


'''
with open(pool1[0],'r') as f:
    data=json.load(f)
with open('data.json', 'w') as f:
    json.dump(data, f,ensure_ascii=False,indent=4)

'''











'''
'aaa_ertong'
'aaa_dangdang/历史传记'
'''