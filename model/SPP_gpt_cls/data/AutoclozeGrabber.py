import pandas as pd
import numpy as np
class AutoclozeGrabber:
    def __init__(self,data_path):
        raise NotImplementedError
    def write_data(self):
        self.train_data[['source','target']].to_csv('./data_train/train.csv')
        self.valid_data[['source','target']].to_csv('./data_train/val.csv')
        self.test_data[['source','target']].to_csv('./data_train/test.csv')
class AutoclozeGrabberCls(AutoclozeGrabber):
    def __init__(self,data_path):
        temp = pd.read_csv(data_path)
        temp['context'] = temp['context'].apply(lambda x: x.replace('<MASK>','[MASK]'))
        temp['source_true'] = temp.apply(lambda x: '[SEP]'.join([x.context,x['True']]),axis =1)
        temp['target_true'] = 1
        temp['source_false'] = temp.apply(lambda x: '[SEP]'.join([x.context,x['False']]),axis =1)
        temp['target_false'] = 0
        temp2 = pd.concat([temp[['context','source_true','target_true']].rename(columns = {'source_true':'source','target_true':'target'}),temp[['context','source_false','target_false']].rename(columns = {'source_false':'source','target_false':'target'})])

        self.data = temp2
        set_c = list(set(self.data.context))
        len_c = len(set_c)
        c_train = set_c[:int(0.8*len_c)]
        c_valid = set_c[int(0.8*len_c):int(0.9*len_c)]
        c_test = set_c[int(0.9*len_c):]
        self.train_data = self.data[self.data.context.isin(c_train)].sample(frac=1).reset_index(drop=True)
        self.valid_data = self.data[self.data.context.isin(c_valid)].sample(frac=1).reset_index(drop=True)
        self.test_data = self.data[self.data.context.isin(c_test)].sample(frac=1).reset_index(drop=True)
 
class AutoclozeGrabberGen(AutoclozeGrabber):
    def __init__(self,data_path):
        temp = pd.read_csv(data_path)
        temp['context'] = temp['context'].apply(lambda x: x.replace('<MASK>','<mask>'))
        temp['source'] = temp.context
        temp['target'] = temp['True']
        self.data = temp
        set_c = list(set(self.data.context))
        len_c = len(set_c)
        c_train = set_c[:int(0.8*len_c)]
        c_valid = set_c[int(0.8*len_c):int(0.9*len_c)]
        c_test = set_c[int(0.9*len_c):]
        self.train_data = self.data[self.data.context.isin(c_train)].sample(frac=1).reset_index(drop=True)
        self.valid_data = self.data[self.data.context.isin(c_valid)].sample(frac=1).reset_index(drop=True)
        self.test_data = self.data[self.data.context.isin(c_test)].sample(frac=1).reset_index(drop=True)

class AutoclozeGrabberTrippleCls(AutoclozeGrabber):
    def __init__(self,data_path):
        temp = pd.read_csv(data_path)
        temp['context'] = temp['context'].apply(lambda x: x.replace('<MASK>','[MASK]'))
        temp['source_true'] = temp.apply(lambda x: '[SEP]'.join([x.context,x['True'],x['False']]),axis =1)
        temp['target_true'] = 0
        temp['source_false'] = temp.apply(lambda x: '[SEP]'.join([x.context,x['False'],x['True']]),axis =1)
        temp['target_false'] = 1
        __len = len(temp)
        sep = int(__len/2.0)
        temp = temp.sample(frac=1).reset_index(drop=True)
        temp2 = pd.concat([temp.iloc[:][['context','source_true','target_true']].rename(columns = {'source_true':'source','target_true':'target'}),temp.iloc[:][['context','source_false','target_false']].rename(columns = {'source_false':'source','target_false':'target'})])
        self.data = temp2
        set_c = list(set(self.data.context))
        len_c = len(set_c)
        c_train = set_c[:int(0.8*len_c)]
        c_valid = set_c[int(0.8*len_c):int(0.9*len_c)]
        c_test = set_c[int(0.9*len_c):]
        self.train_data = self.data[self.data.context.isin(c_train)].sample(frac=1).reset_index(drop=True)
        self.valid_data = self.data[self.data.context.isin(c_valid)].sample(frac=1).reset_index(drop=True)
        self.test_data = self.data[self.data.context.isin(c_test)].sample(frac=1).reset_index(drop=True)
 


class ClozeRgrabber(AutoclozeGrabber):
    def merge_tf(self,temp):
        return pd.concat([temp.iloc[:][['context','source_true','target_true']].rename(columns = {'source_true':'source','target_true':'target'}),temp.iloc[:][['context','source_false','target_false']].rename(columns = {'source_false':'source','target_false':'target'})])

    def __init__(self,train_path,valid_path,test_path):
        self.train_data = pd.read_csv(train_path)
        self.valid_data = pd.read_csv(valid_path)
        self.test_data = pd.read_csv(test_path)
        print(len(self.train_data))
        for temp in [self.train_data,self.valid_data,self.test_data]:
            temp['context'] = temp['context'].apply(lambda x: x.replace('<MASK>','[MASK]'))
            temp['source_true'] = temp.apply(lambda x: '[SEP]'.join([x.context,x['True'],x['False']]),axis =1)
            temp['target_true'] = 0
            temp['source_false'] = temp.apply(lambda x: '[SEP]'.join([x.context,x['False'],x['True']]),axis =1)
            temp['target_false'] = 1
        self.train_data = self.merge_tf(self.train_data)
        self.valid_data = self.merge_tf(self.valid_data)
        self.test_data = self.merge_tf(self.test_data)
        print(len(self.train_data))
class outgengrabber(AutoclozeGrabber):
    def __init__(self,train_path,valid_path,test_path):
        self.train_data = pd.read_csv(train_path)
        self.valid_data = pd.read_csv(valid_path)
        self.test_data = pd.read_csv(test_path)
        for temp in [self.train_data,self.valid_data,self.test_data]:
            temp['source'] = temp['keywords']
            temp['target'] = temp['story']
            temp['data'] = temp.apply(lambda x: x.source.replace(',','#')+'[SEP]'+x.target,axis = 1)

class SPPgrabber(AutoclozeGrabber):
    def __init__(self,train_path,valid_path,test_path):
        import json
        self.train_data = pd.DataFrame(json.load(open(train_path,'r'))['data'])
        self.valid_data =pd.DataFrame(json.load(open(valid_path,'r'))['data'])
        self.test_data = pd.DataFrame(json.load(open(test_path,'r'))['data'])
        #import pdb;pdb.set_trace()
        for temp in [self.train_data,self.valid_data,self.test_data]:

            temp['source'] = temp['text']
            temp['target'] = temp['label']

 
if __name__ == '__main__':
    graber = AutoclozeGrabberTrippleCls('../../../40-pn-dataset-auto.csv')
    path = '/home/fengzhuoer/data/SPP'
    graber = SPPgrabber(path + '/train.json',path + '/val.json',path + '/test.json')
    graber.write_data()
