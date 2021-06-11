import pandas as pd
data = pd.read_csv('99-dataset/ClozeR/ClozeR_valid.csv')

data = data.sample(frac=1).reset_index(drop=True)


t = data.iloc[:50][['context','True','False']].rename(columns = {'True':'1','False':'0'})
t['label'] = 1
f = data.iloc[50:100][['context','False','True']].rename(columns = {'False':'1','True':'0'})
f['label'] = 0
print(t.columns,f.columns)

d = pd.concat([f,t]).sample(frac=1).reset_index(drop=True)

d.to_csv('93-annotate-100-clozeR-valid.csv')


