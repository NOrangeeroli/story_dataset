from itertools import product
import pandas as pd
def if_rep(x):
    s = x.split(',')
    for a,b in product(s,s):
        if a!= b and a in b:
            return True
    return False
def report_rep_num(path):
    df  = pd.read_csv(path)
    df['if'] = df['keywords'].apply(if_rep)
    return df['if'].sum()#,df[df['if']].keywords.iloc[:10]

p = '99-dataset/OutGen/OutGen_'
path = p + 'train.csv'
print(path, report_rep_num(path),'/',len(pd.read_csv(path)))
path = p + 'valid.csv'
print(path, report_rep_num(path),'/',len(pd.read_csv(path)))
path = p + 'test.csv'
print(path, report_rep_num(path),'/',len(pd.read_csv(path)))
