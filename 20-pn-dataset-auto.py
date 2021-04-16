from bert_score import BERTScorer
import random
import pandas as pd
from utils import split_by_fullstop
from tools import start_debugger_on_exception
import torch
start_debugger_on_exception()
scorer = BERTScorer(lang="zh", rescale_with_baseline=True)
data_file = 'annotateddata/batch1.csv'
df = pd.read_csv(data_file)
# import pdb;pdb.set_trace()
stories = list(df.RESULT.dropna())
stories_split = [split_by_fullstop(x) for x in stories]
refs_pre = [x for y in stories_split for x in y if len(x)>0]
stories_split_select = [random.randint(0,len(x)-1) for x in stories_split]
stories_sentencesample = [x[y] for x,y in zip(stories_split,stories_split_select)]

stories_context = [] 
for ss,sss in zip(stories_split,stories_split_select):
    ss[sss] = '<MASK>'
    stories_context.append(ss)
stories_context = [''.join(x) for x in stories_context]  
positive_samples = [(x,y,True) for x,y in zip(stories_context,stories_sentencesample)]
cands_pre = stories_sentencesample
len_refs = len(refs_pre)
len_cands = len(cands_pre)
cands = [x for x in cands_pre for i in range(len_refs)]

refs = refs_pre*len_cands
# print(refs)
# print(cands)
P, R, F1 = scorer.score(cands, refs)
print(R.reshape(len_cands,len_refs))
R=R.reshape(len_cands,len_refs)
bestmatch = torch.topk(R,2,dim=1).indices[:,1]
negative_samples = [(x,refs_pre[y],False) for x,y in zip(stories_context,bestmatch)]
import pdb;pdb.set_trace()
samples = negative_samples+positive_samples
result_df = pd.DataFrame(samples, columns = ['context','keysen','label'])
print(result_df.head())
result_df.to_csv('data/autocloze.csv',encoding="utf_8_sig")
result_df.to_excel('data/autocloze.xlsx',encoding="utf_8_sig")


# print(R.reshape(2,3))

#import pdb;pdb.set_trace()
