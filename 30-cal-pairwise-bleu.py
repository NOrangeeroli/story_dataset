import pandas as pd
import nltk
def read_annotated_data():
    batch1 = pd.read_csv('annotateddata/batch1.csv',encoding="utf_8_sig")[['title','RESULT']].dropna(subset = ['RESULT']).rename(columns = {'RESULT':'story'})
    batch2 = pd.read_csv('annotateddata/batch2.csv',encoding="utf_8_sig")
    batch2 = batch2[batch2.reject!='1'][['title','result']].dropna(subset=['result']).rename(columns = {'result':'story'})

    r = pd.concat([batch1, batch2])
    r = r.drop_duplicates(subset = ['story'])
    r['story'] = r['story'].apply(lambda x: x.replace('。。','。').strip())
    return r

r = read_annotated_data()
r.to_csv('30-raw_stories.csv',encoding="utf_8_sig")
result_list = r.story.values.tolist()
bleus = {}
for i,a in enumerate(result_list):
    for j, b in enumerate(result_list):
        print(i,j)
        if i<j:
            refs = [b]
            references = [list(x) for x in refs]
            hypothesis = list(a)
            bleus[a,b] = nltk.translate.bleu_score.sentence_bleu(references, hypothesis)

#import pdb;pdb.set_trace()
ab = [(a,b) for a,b in bleus.keys()]
v = [bleus[x] for x in ab]
a = [x for x,y in ab]
b = [y for x,y in ab]
df_result = pd.DataFrame({'a':a,'b':b,'bleu':v})
df_result.sort_values(by = 'bleu').tail(500).to_csv('30-highduplicate.csv',encoding="utf_8_sig")
import pdb;pdb.set_trace()
