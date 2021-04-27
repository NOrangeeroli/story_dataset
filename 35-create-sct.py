import pandas as pd
from tools import start_debugger_on_exception
import numpy as np
stories= pd.read_csv('34-filter-trivial-ending.csv')
import  random
from utils import split_by_fullstop
start_debugger_on_exception()
def get_anocontext_randmid(x):
    s = split_by_fullstop(x)
    if len(s)<=2:
        return False
    sample_index = random.randint(1,len(s)-2)
    s[sample_index]='【'+s[sample_index]+'】'
    return ''.join(s)
def get_anocontext_end(x):
    if x.high_end_bleu:
        return False
    else:
        s = split_by_fullstop(x.story)
        s[-1]='【'+s[-1]+'】'
        return ''.join(s)
stories['mid_cloze'] = stories.story.apply(lambda x: get_anocontext_randmid(x))
stories['end_cloze'] = stories.apply(lambda x: get_anocontext_end(x),axis = 1)
midcloze = stories[['title','mid_cloze']].rename(columns = {'mid_cloze':'cloze'})
midcloze = midcloze[midcloze.cloze != False]
endcloze = stories[['title','end_cloze']].rename(columns = {'end_cloze':'cloze'})
endcloze = endcloze[endcloze.cloze != False]
cloze = pd.concat([endcloze,midcloze]).sort_values(by = 'title')
cloze['REJECT']=np.nan
cloze['FIND_SENTENCE']=np.nan
cloze['FILL'] = np.nan
cloze['STRATEGY'] = np.nan
cloze.to_csv('data/sct_tobeanoed.csv',encoding="utf_8_sig")




