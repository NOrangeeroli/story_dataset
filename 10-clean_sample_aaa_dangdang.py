import pandas as pd
import json
import glob,os,sys
from utils import *

SENLEN = 5
cats = [ '玄幻小说', '科幻小说', \
'恐怖悬疑', '青春校园', '玄幻奇幻', '历史传记', \
'武侠小说', '武侠仙侠', '文学名著']
cats = [ '历史传记']

def split_paragraphs(string):
    string = string.replace('|','')
    s = split_by_dialogquote(string)
    s = [complete_quote(x) for x in s if len(x)>0]
    
    
    
    s = [x for x in s if not is_dialogquote(x)]
    s = [strip_incomplete_sentence(x)  for x in s if len(x)>0]
    s = [x for x in s if count_clauses(x)>3 and paragraph_sanity_check(x)]

    return list(set(s))
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
pool1=glob.glob(data_path+'/aaa_dangdang/*')
pool1 = [x for x in pool1 if x.split('_')[2] in cats]
# print(pool1)
sentences_count=nested_dict()
backup = {}
# sentences_split=nested_dict()
paragraphs_split = nested_dict()
for i,fn in enumerate(pool1):
    with open(fn,'r') as f:
        data=json.load(f)
    print(i/float(len(pool1)))
    backup[fn] = data
    # sentences_count[data['title']]=[is_dialogquote_incomplete(x['text']) for x in data['chapter']]
    paragraphs_split[fn]=sum([split_paragraphs(c['text']) for c in data['chapter']],[])
sentences_split= {k:sum([split_sentences(p) for p in paragraphs_split[k]],[]) for k in paragraphs_split.keys()}   
with open('aaa_dangdang.json','w') as f:
    json.dump(sentences_split, f,ensure_ascii=False,indent=4)

sentence_length= [count_clauses(y)  for k in sentences_split.keys() for y in sentences_split[k]]    
# print(sum([sum(x) for x in sentences_count.values()]))
import pdb;pdb.set_trace()

    
    


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