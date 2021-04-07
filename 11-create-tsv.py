from __future__ import unicode_literals
import json
import pandas as pd
from utils import *


def create_sample_from_rawentry(stringa,stringb):
    import random
    sa = split_by_fullstop(stringa)
    sb = split_by_fullstop(stringb)
    if len(sa)<=1 or len(sb)<=1:
        return None
    else:
        masklocationa = random.randint(1,len(sa))
        masklocationb = random.randint(1,len(sb))
        # print(len(sa),masklocationa)
        seqb_false = sb[masklocationb-1]
        seqb_true = sa[masklocationa-1]
        sa[masklocationa-1] = 'MASK'
        seqa = ''.join(sa)
        return [{'label':True,'seqa':seqa,'seqb':seqb_true}, \
        {'label':False,'seqa':seqa,'seqb':seqb_false}]
def create_samples_from_entries(entries):
    import random
    samples=[]
    for i in range(4):
        random.shuffle(entries)
        for j in range(int(len(entries)/2)):
            temp = create_sample_from_rawentry(entries[2*j],entries[2*j+1])
            if temp is not None:
                samples += temp
    return samples
def seperate_train_dev(list_):
    sep1 = int(len(list_)/5*4/10)
    sep2 = int(len(list_)/5*5/10)
    return list_[:sep1], list_[sep1:sep2]

    

def write_tsv(sample_list,fn):
    import csv
    with open(fn, mode='w') as f:
        writer = csv.writer(f, delimiter='\t')
        # writer.writerow(['label','seqa','seqb'])
        for sample in sample_list:
            label, seqa,seqb = sample['label'],sample['seqa'],sample['seqb']
            writer.writerow([label, seqa,seqb])





jfiles=['aaa_ertong.json']

rawdata = {}
for fn in jfiles:
    with open(fn,'r') as f:
        rawdata.update(json.load(f))
print (rawdata.keys())
train_keys, dev_keys = seperate_train_dev(list(rawdata.keys()))
train_entries = [x for k in train_keys for x in rawdata[k] ]
dev_entries = [x for k in dev_keys for x in rawdata[k] ]
train_samples = create_samples_from_entries(train_entries)
dev_samples = create_samples_from_entries(dev_entries)
# df_train_samples  = pd.DataFrame(train_samples)[['label','seqa','seqb']]
# df_dev_samples  = pd.DataFrame(dev_samples)[['label','seqa','seqb']]

write_tsv(train_samples,'train.tsv')

write_tsv(dev_samples,'dev.tsv')


import pdb;pdb.set_trace()
