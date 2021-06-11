import spacy
import pandas as pd
from collections import Counter
nlp = spacy.load("zh_core_web_trf")
clozer = pd.concat([pd.read_csv('99-dataset/ClozeR/ClozeR_'+x+'.csv') for x in ['train','test','valid']])
PC = pd.concat([pd.read_csv('99-dataset/PC/PC_'+x+'.csv') for x in ['train','test','valid']])
PC['story']  = PC.apply(lambda x: x.context.replace('<MASK>',x['True']),axis = 1)
clozer['story'] = clozer.apply(lambda x: x.context.replace('<MASK>',x['True'])+x['False'],axis = 1)
def write_topic(name,stories):
    l = len(stories) 
    Nouns = []
    for i,( s,f) in enumerate(stories.items()):
        print (i,'/',l)
        doc = nlp(s)
        for w,c in [(w.text, w.pos_) for w in doc]:
            if c == 'NOUN':
                Nouns+=[w]*f
    cn = Counter(Nouns)
    top100_nouns = pd.DataFrame(cn,index = [0]).T.sort_values(ascending = False,by = 0).iloc[:100]
    top100_nouns.to_csv('90-'+name + '-topic-analysis.csv')
write_topic('clozer',Counter(clozer['story'].tolist()))
write_topic('PC',Counter(PC['story'].tolist()))
import pdb;pdb.set_trace()
