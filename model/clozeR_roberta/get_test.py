f = open('log','r')
lis = []
for x in f.readlines():
    s =x.split() 
    dic = {}
    for i in range(int(len(s)/2)):
        dic[s[2*i]]=s[i*2+1]
    lis.append(dic)
import pandas as pd
df = pd.DataFrame(lis)
r = df[df['val_acc'] == df['val_acc'].max()][['test_acc','val_acc']]
print(r)
import pdb;pdb.set_trace()
