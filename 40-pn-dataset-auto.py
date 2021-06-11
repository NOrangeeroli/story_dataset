from bert_score import BERTScorer
import random
import pandas as pd
from utils import split_by_fullstop
from tools import start_debugger_on_exception
import torch
import copy
start_debugger_on_exception()
def gen_samples():
        
    scorer = BERTScorer(lang="zh", rescale_with_baseline=True)
    data_file = '32-deduplicate-story.csv'
    df = pd.read_csv(data_file)
    # import pdb;pdb.set_trace()
    stories = list(df.story.dropna())
    stories_split = [split_by_fullstop(x) for x in stories]
    stories_split_select = [random.randint(0,len(x)-1) for x in stories_split]
    stories_sentencesample = [x[y] for x,y in zip(stories_split,stories_split_select)]
    stories_split_copy = copy.deepcopy(stories_split)
    stories_context = [] 
    for ss,sss in zip(stories_split_copy,stories_split_select):
        ss[sss] = '<MASK>'
        stories_context.append(ss)
    stories_context = [''.join(x) for x in stories_context]  
    positive_samples = [(x,y,True) for x,y in zip(stories_context,stories_sentencesample)]
    cands = stories_sentencesample
    assert len(cands)==len(stories_split)
    refs = []
    for i,cand in enumerate(cands):
        refs.append([x for j,y in enumerate(stories_split) for x in y  if len(x)>0 and j!=i])
    bestmatch = []
    print(len(cands))
    for i,(c, ref) in enumerate(zip(cands, refs)):
        print(i,'th candidate...')
        cand = [c]*len(ref)
        import pdb;pdb.set_trace()
        P, R, F1 = scorer.score(cand, ref)
        bestmatch.append(int(torch.argmax(R)))
    negative_samples = [(x,y[z],False) for x,y,z in zip(stories_context,refs,bestmatch)]
    return [(x,w,y[z]) for x,y,z,w in zip(stories_context,refs,bestmatch,stories_sentencesample)]
samples = gen_samples()
result_df = pd.DataFrame(samples, columns = ['context','True','False'])
print(result_df.head())
result_df.to_csv('40-pn-dataset-auto.csv',encoding="utf_8_sig")


# print(R.reshape(2,3))

#import pdb;pdb.set_trace()
