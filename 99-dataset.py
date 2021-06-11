import pandas as pd
from utils import split_by_fullstop
import random
from Rake_For_Chinese.src.Rake import Rake
from rake import rakezh
from utils import get_word_len
from rake import rakezh
from tools import start_debugger_on_exception
start_debugger_on_exception()
random.seed(1)
def gen_ClozeR():
    df = pd.read_csv('91-adjust_bias.csv')
    df['story'] = df.apply(lambda x: x['context'].replace('<MASK>',x['True']),axis = 1)
    stories = sorted(list(set(df['story'].tolist())))
    random.shuffle(stories)
    stories_in_train = stories[:int(len(stories)*0.55)]
    stories_in_valid = stories[int(len(stories)*0.55):int(len(stories)*0.66)]
    stories_in_test = stories[int(len(stories)*0.66):]
    train = df[df['story'].isin(stories_in_train)]
    valid = df[df['story'].isin(stories_in_valid)]
    test = df[df['story'].isin(stories_in_test)]
    train[['context','True','False','strategy']].to_csv('99-dataset/ClozeR/ClozeR_train.csv')
    valid[['context','True','False','strategy']].to_csv('99-dataset/ClozeR/ClozeR_valid.csv')
    test[['context','True','False','strategy']].to_csv('99-dataset/ClozeR/ClozeR_test.csv')

def gen_ClozeT():
    pass

def gen_PC():
    vt = pd.read_csv('91-adjust_bias.csv')
    vt = vt[['context','True']]
    vt['story'] = vt.apply(lambda x: x['context'].replace('<MASK>',x['True']),axis = 1)
    df = pd.read_csv('32-deduplicate-story.csv')
    dict_sct = []
    for story in df.story:
        for s in split_by_fullstop(story):
            dict_sct.append({'story':story,'context':story.replace(s,'<MASK>'),'True':s })
    train= pd.DataFrame(dict_sct)
    train = train[~train.story.isin(vt['story'].tolist())]
    train[['context','True']].to_csv('99-dataset/PC/PC_train.csv')
    stories_in_vt = sorted(list(set(vt['story'].tolist())))
    random.shuffle(stories_in_vt)
    stories_in_valid = stories_in_vt[:int(len(stories_in_vt)*0.5)]
    stories_in_test = stories_in_vt[int(len(stories_in_vt)*0.5):]
    valid = vt[vt.story.isin(stories_in_valid)]
    test = vt[vt.story.isin(stories_in_test)]
    valid.to_csv('99-dataset/PC/PC_valid.csv')
    test.to_csv('99-dataset/PC/PC_test.csv')
def gen_OutGen():
    infile = '32-deduplicate-story.csv'
    stories = pd.read_csv(infile)
    def get_keywords(x):
        result = rakezh(x)
        result = [(x,y) for x,y in result if get_word_len(x)<8]
        result = sorted(result, key=lambda x: len(x[0]),reverse=True)
        r = []
        for w,f in result:
            skip = False
            for ref,_ in r:
                if w in ref:
                    #import pdb;pdb.set_trace()
                    skip = True
                    break
            if skip:
                #import pdb;pdb.set_trace()
                continue
            r.append((w,f))
        r = sorted([(x,y) for x,y in r],key = lambda x: x[1],reverse=True)
        if len(r)>8:
            r = r[:8]
        r = [x for x,y in r]
        assert len(r)>0
        return ','.join(r)
    import pdb;pdb.set_trace()
    
    stories['keywords']= stories.apply(lambda x: get_keywords(x.story),axis = 1)
    stories = stories[['title','story','keywords']].dropna()
    stories = stories[stories['keywords'].apply(lambda x: len(x)>0)]
    stories_set = sorted(list(set(stories['story'].tolist())))
    random.shuffle(stories_set)  
    stories_in_train = stories_set[:int(len(stories_set)*0.6)]
    stories_in_valid = stories_set[int(len(stories_set)*0.6):int(len(stories_set)*0.7)]
    stories_in_test = stories_set[int(len(stories_set)*0.7):]
    train = stories[stories['story'].isin(stories_in_train)]
    valid = stories[stories['story'].isin(stories_in_valid)]
    test = stories[stories['story'].isin(stories_in_test)]
    train.to_csv('99-dataset/OutGen/OutGen_train.csv')
    valid.to_csv('99-dataset/OutGen/OutGen_valid.csv')
    test.to_csv('99-dataset/OutGen/OutGen_test.csv')

#gen_ClozeR()
#gen_PC()
gen_OutGen()
