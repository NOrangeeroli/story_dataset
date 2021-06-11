import pandas as pd
import numpy as np
class AutoclozeGrabber:
    def __init__(self,data_path):
        raise NotImplementedError
    def write_data(self):
        with open('data_train/train.source','w') as f:
            for index,row in self.train_data.iterrows():
                f.write(row['source']+'\n')
        with open('data_train/train.target','w') as f:
            for index,row in self.train_data.iterrows():
                f.write(str(row['target'])+'\n')
        with open('data_train/val.target','w') as f:
            for index,row in self.valid_data.iterrows():
                f.write(str(row['target'])+'\n')
        with open('data_train/val.source','w') as f:
            for index,row in self.valid_data.iterrows():
                f.write(row['source']+'\n')
        with open('data_train/test.source','w') as f:
            for index,row in self.test_data.iterrows():
                f.write(row['source']+'\n')
        with open('data_train/test.target','w') as f:
            for index,row in self.test_data.iterrows():
                f.write(str(row['target'])+'\n')
class AutoclozeGrabberCls(AutoclozeGrabber):
    def __init__(self,data_path):
        temp = pd.read_csv(data_path)
        temp['context'] = temp['context'].apply(lambda x: x.replace('<MASK>','<mask>'))
        temp['source_true'] = temp.apply(lambda x: '<sep>'.join([x.context,x['True']]),axis =1)
        temp['target_true'] = 1
        temp['source_false'] = temp.apply(lambda x: '<sep>'.join([x.context,x['False']]),axis =1)
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

class AutoclozeGrabberClozeR(AutoclozeGrabber):
    def merge_tf(self,temp):
        return pd.concat([temp.iloc[:][['context','source_true','target_true']].rename(columns = {'source_true':'source','target_true':'target'}),temp.iloc[:][['context','source_false','target_false']].rename(columns = {'source_false':'source','target_false':'target'})])

    def __init__(self,train_path,valid_path,test_path):
        self.train_data = pd.read_csv(train_path)
        self.valid_data = pd.read_csv(valid_path)
        self.test_data = pd.read_csv(test_path)
        print(len(self.train_data))
        for temp in [self.train_data,self.valid_data,self.test_data]:
            temp['context'] = temp['context'].apply(lambda x: x.replace('<MASK>','<mask>'))
            temp['source_true'] = temp.apply(lambda x: '<sep>'.join([x.context,x['True'],x['False']]),axis =1)
            temp['target_true'] = 0
            temp['source_false'] = temp.apply(lambda x: '<sep>'.join([x.context,x['False'],x['True']]),axis =1)
            temp['target_false'] = 1
        self.train_data = self.merge_tf(self.train_data)
        self.valid_data = self.merge_tf(self.valid_data)
        self.test_data = self.merge_tf(self.test_data)
        print(len(self.train_data))



if __name__ == '__main__':
    graber = AutoclozeGrabberGen('../../../40-pn-dataset-auto.csv')
    path = '../../../99-dataset/ClozeR/'
    graber = AutoclozeGrabberClozeR(path + 'ClozeR_train.csv',path + 'ClozeR_valid.csv',path + 'ClozeR_test.csv')
    graber.write_data()
