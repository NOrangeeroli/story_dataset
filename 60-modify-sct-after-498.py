import pandas as pd
from utils import split_by_pair,split_by_fullstop
df = pd.read_csv('annotateddata/05-14anticomm-top500.csv')

rest = df.iloc[498:]
rest = rest['标题,给定的故事,REJECT,找出其中可以使用常识推断的句子,将标出的句子改写为一个与上下文内容相关但违反常识的句子,改写策略'.split(',')]
rest['temp'] = rest['给定的故事'].apply(lambda x: split_by_pair(x,left = '【',right = '】'))
rest['temp0'] = rest['temp'].apply(lambda x: split_by_fullstop(x[0]))
rest['temp2'] = rest['temp'].apply(lambda x: split_by_fullstop(x[2]))
rest['temp'] = rest['temp'].apply(lambda x: [x[1]])
rest['temp'] = rest.apply(lambda x: [k for k in x.temp0+x.temp+x.temp2 if len(k)>0],axis = 1)
rest['temp'] = rest['temp'].apply(lambda x: ''.join(['{'+str(i)+'}'+ k for i,k in enumerate(x)]))
rest['给定的故事'] = rest['temp']
rest = rest['标题,给定的故事,REJECT,找出其中可以使用常识推断的句子,将标出的句子改写为一个与上下文内容相关但违反常识的句子,改写策略'.split(',')]
rest.to_csv('data/sct_tobeanoed_after498_modify.csv',encoding="utf_8_sig")
import pdb;pdb.set_trace()

