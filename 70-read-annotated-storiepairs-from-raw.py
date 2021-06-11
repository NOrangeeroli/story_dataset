import pandas as pd
import numpy as np
from tools import start_debugger_on_exception
import re
start_debugger_on_exception()
def get_results_batch1(x):
    r = {}
    r['story'] = x['给定的故事'].replace('【','').replace('】','')
    r['title'] = x['标题']
    if x.Mark!= 1:
        r['True'] = x['找出其中可以使用常识推断的句子']
        r['False'] = x['将标出的句子改写为一个与上下文内容相关但违反常识的句子']
        r['strategy'] = x['改写策略']
    else:
        if x['重新选择句子'] is not np.nan:
            r['True'] = x['重新选择句子']
            r['False'] = x['改写']
            r['strategy'] = x['改写策略']
        else:
            r['True'] = x['找出其中可以使用常识推断的句子']
            r['False'] = x['改写']
            r['strategy'] = x['改写策略']
    return r

def get_results_batch2(x):
    r = {}
    r['story'] = re.sub(r'\{[0-9]+\}',r'',x['给定的故事'])
    r['story']= r['story'].replace('【','').replace('】','')
    r['title'] = x['标题']
    if x.Mark!= 1:
        r['True'] = x['找出其中可以使用常识推断的句子']
        r['False'] = x['将标出的句子改写为一个与上下文内容相关但违反常识的句子']
        r['strategy'] = x['改写策略']
    else:
        if x['select(correction)'] is not np.nan:
            r['True'] = x['select(correction)']
            r['False'] = x['rewrite(correction)']
            
        else:
            r['True'] = x['找出其中可以使用常识推断的句子']
            r['False'] = x['rewrite(correction)']
        if str(x['strategy(correction)']) !='nan':
            r['strategy'] =  x['strategy(correction)']
        else:
            r['strategy'] = x['改写策略']
    #if r['False'] == '每当它看到猫叫的时候，就威胁猫闭嘴。':
    #    import pdb;pdb.set_trace()
    return r
 
def get_results_batch4(x):
    r = {}
    r['story'] = re.sub(r'\{[0-9]+\}',r'',x['给定的故事'])
    r['story']= r['story'].replace('【','').replace('】','')
    r['title'] = x['标题']
    r['True'] = x['找出其中可以使用常识推断的句子']
    r['False'] = x['将标出的句子改写为一个与上下文内容相关但违反常识的句子']
    r['strategy'] = x['改写策略']
    return r



batch1 = pd.read_csv('annotateddata/05-14anticomm-top500.csv',dtype = {'Mark':float}).head(500)
batch1 = batch1[batch1['REJECT']!=1]
batch1 = batch1[batch1['Reject']!=1]
batch1 = batch1.apply(lambda x: get_results_batch1(x), axis = 1,result_type  = 'expand')


batch2 = pd.read_csv('annotateddata/0518正式数据-500-1000.csv',dtype = {'Mark':float}).head(503)
batch2 = batch2[batch2['REJECT']!=1]
batch2 = batch2[batch2['Reject(correction)']!=1]
batch2 = batch2.apply(lambda x: get_results_batch2(x), axis = 1,result_type  = 'expand')

batch3 = pd.read_csv('annotateddata/0521正式数据-1001-1500.csv',dtype = {'Mark':float}).head(500)
batch3 = batch3[batch3['REJECT']!=1]
batch3 = batch3[batch3['Reject(correction)']!=1]
batch3 = batch3.apply(lambda x: get_results_batch2(x), axis = 1,result_type  = 'expand')

batch4 = pd.read_csv('annotateddata/0604修改.csv',dtype = {'Mark':float}).head(500)
def get_diff_cl(x):
    if x['REJECT']==1:
        return 0
    else:
        return len(x['将标出的句子改写为一个与上下文内容相关但违反常识的句子'].split('，'))-len(x['找出其中可以使用常识推断的句子'].split('，'))
def get_diff_wr(x):
    if x['REJECT']==1:
        return 1
    else:
        return len(x['将标出的句子改写为一个与上下文内容相关但违反常识的句子'])/len(x['找出其中可以使用常识推断的句子'])
batch4['diff_cl'] = batch4.apply(get_diff_cl,axis = 1)
batch4['diff_wr'] = batch4.apply(get_diff_wr,axis = 1)
batch4['分局数量不符合要求'] = (batch4['diff_cl']>1)|(batch4['diff_cl']<-1)
batch4['句子长度不符合要求'] = (batch4['diff_wr']>2)|(batch4['diff_wr']<0.5)
batch4.to_csv('0602feedback.csv')
batch4 = batch4[batch4['REJECT']!=1]

#batch4 = batch4[batch4['Reject(correction)']!=1]
batch4 =batch4.apply(lambda x: get_results_batch4(x), axis = 1,result_type  = 'expand')

final = pd.concat([batch4])
final = final.dropna()
final['assert']= final.apply(lambda x: x['True'] in x.story,axis = 1)
assert final['assert'].all()
final['context'] = final.apply(lambda x: x['story'].replace(x['True'],'<MASK>'),axis = 1)
final = final.drop_duplicates(subset = ['context'])
def mod_strategy(x):
    if x == '·1':
        return 1
    elif x =='1':
        return 1
    elif x == '2':
        return 2
    elif isinstance(x,int) or isinstance(x,float):
        return int(x)
    else:
        assert False
final['strategy']=final['strategy'].apply(mod_strategy)
final[['context','True','False','strategy']].to_csv('70-read-annotated-storiepairs-from-raw.csv')

import pdb;pdb.set_trace()
