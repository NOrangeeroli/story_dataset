import spacy
import json
import pandas as pd
nlp = spacy.load("zh_core_web_trf")
path = '/home/fengzhuoer/data/SPP'

train_data = pd.DataFrame(json.load(open(path + '/train.json','r'))['data'])
test_data = pd.DataFrame(json.load(open(path + '/test.json','r'))['data'])
valid_data = pd.DataFrame(json.load(open(path + '/val.json','r'))['data'])
data = pd.concat([train_data,test_data,valid_data])
data['story'] = data.text.apply(lambda x: x.replace('[P]','').replace('[SEP]','').replace('[CLS]',''))

Nouns = []
Verbs = []
Hoka = []
l = len(data['story'].tolist())
for i,s in enumerate(data['story'].tolist()):
    print(i,'/',l)
    doc = nlp(s)
    for w,c in [(w.text, w.pos_) for w in doc]:
        if c == 'NOUN':
            Nouns.append(w)
        elif c == 'VERB':
            Verbs.append(w)
        else:
            Hoka.append(w)
from collections import Counter
cn = Counter(Nouns)
ch = Counter(Hoka)
top50_hoka = pd.DataFrame(ch,index = [0]).T.sort_values(ascending = False,by = 0)
top50_nouns = pd.DataFrame(cn,index = [0]).T.sort_values(ascending = False,by = 0).iloc[:100]
top50_nouns.to_csv('901-SPP-topic-analysis.csv')
import pdb;pdb.set_trace()
