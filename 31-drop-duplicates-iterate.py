import pandas as pd
import nltk
import numpy as np
import glob
def read_annotated_data():
    batch1 = pd.read_csv('annotateddata/batch1.csv',encoding="utf_8_sig")[['title','RESULT']].dropna(subset = ['RESULT']).rename(columns = {'RESULT':'story'})
    batch2 = pd.read_csv('annotateddata/batch2.csv',encoding="utf_8_sig")
    batch2 = batch2[batch2.reject!='1'][['title','result']].dropna(subset=['result']).rename(columns = {'result':'story'})

    r = pd.concat([batch1, batch2])
    r = r.drop_duplicates(subset = ['story'])
    r['story'] = r['story'].apply(lambda x: x.replace('。。','。').strip())
    return r

r = pd.read_csv('analysis/highduplicate.csv',encoding="utf_8_sig")
manual_csvs =glob.glob('31-drop-duplicates*manual.csv') 
df_to_delete = pd.concat([pd.read_csv(c,encoding="utf_8_sig") for c in manual_csvs])
list_to_delete = df_to_delete[df_to_delete['Unnamed: 0']==3000]['0'].tolist()
r = r[~r['a'].isin(list_to_delete)]
r = r[~r['b'].isin(list_to_delete)].sort_values(by = 'bleu').tail(200)
temp = []
for row in r.iterrows():
    a = row[1].a
    b = row[1].b
    bleu = row[1].bleu
    if len(temp)==0:
        temp.append([[a,b],[bleu]])
    br = False
    for tlist in temp:
        if a in tlist[0] and not b in tlist[0]:
            tlist[0].append(b)
            tlist[0] = list(set(tlist[0]))
            tlist[1].append(bleu)
            br = True
            break
        elif not a in tlist[0] and b in tlist[0]:
            tlist[0].append(a)
            tlist[0] = list(set(tlist[0]))
            tlist[1].append(bleu)
            br = True
            break
        elif a in tlist[0] and b in tlist[0]:
            br = True
            break
        else:
            pass
    if not br:
        temp.append([[a,b],[bleu]])
temp = [[x,np.mean(y)] for [z,y] in temp for x in z]
pd.DataFrame(temp).to_csv('31-drop-duplicates-2.csv')
#manually annotated duplicates
import pdb;pdb.set_trace()
