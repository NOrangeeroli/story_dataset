import pandas as pd
import jieba
from utils import split_by_fullstop

from tools import start_debugger_on_exception
start_debugger_on_exception()
def get_word_len(s):
    return len(list(jieba.cut(s, cut_all=False)))
def get_sentence_len(s):
    return len(split_by_fullstop(s))
def get_word_list(s):
    return list(jieba.cut(s, cut_all=False))
def get_word_len_outline(s):
    l = s.split(',')
    lens = [get_word_len(w) for w in l]
    return sum(lens)
def if_mask_end(x):
    s = x.split('<MASK>')
    if s[-1] == '':
        return True
    else:
        return False
def clozeR_stats():
    train = pd.read_csv('99-dataset/ClozeR/ClozeR_train.csv')
    valid = pd.read_csv('99-dataset/ClozeR/ClozeR_valid.csv')
    test = pd.read_csv('99-dataset/ClozeR/ClozeR_test.csv')
    train_text = train.apply(lambda x: x['context'].replace('<MASK>',x['True']),axis = 1)
    valid_text = valid.apply(lambda x: x['context'].replace('<MASK>',x['True']),axis = 1)
    test_text = test.apply(lambda x: x['context'].replace('<MASK>',x['True']),axis = 1)
    train_alternetive = train['False']
    valid_alternetive = valid['False']
    test_alternetive = test['False']
    for name in ["train","valid","test"]:
        vocab = []
        DF = eval(name)
        DF['if_end'] = DF['context'].apply(if_mask_end)
        print(name,'Ending Reasoning', len(DF[DF.if_end==True]) )
        print(name,'Abductive Reasoning', len(DF[DF.if_end==False]) )
        for cat in ["text","alternetive"]:
            df = eval(name+"_"+cat)

            word_len = df.apply(get_word_len)
            sentence_len = df.apply(get_sentence_len)
            vocab+=df.apply(get_word_list).sum()
            print('word sentence len',name,cat,word_len.mean(),sentence_len.mean())
        print('vocab size',name,len(set(vocab)))
def PC_stats():
    train = pd.read_csv('99-dataset/PC/PC_train.csv')
    valid = pd.read_csv('99-dataset/PC/PC_valid.csv')
    test = pd.read_csv('99-dataset/PC/PC_test.csv')
    train_input =  train['context'].apply(lambda x: x.replace('<MASK>',''))
    train_output = train['True'].apply(lambda x: x.replace('<MASK>',''))
    valid_input =  valid['context'].apply(lambda x: x.replace('<MASK>',''))
    valid_output = valid['True'].apply(lambda x: x.replace('<MASK>',''))
    test_input =  test['context'].apply(lambda x: x.replace('<MASK>',''))
    test_output = test['True'].apply(lambda x: x.replace('<MASK>',''))
    for name in ["train","valid","test"]:
        vocab = []
        for cat in ["input","output"]:
            df = eval(name+"_"+cat)
            word_len = df.apply(get_word_len)
            sentence_len = df.apply(get_sentence_len)
            vocab+=df.apply(get_word_list).sum()
            print('word sentence len',name,cat,word_len.mean(),sentence_len.mean())
        print('vocab size',name,len(set(vocab)))

def OutGen_stats():
    train = pd.read_csv('99-dataset/OutGen/OutGen_train.csv')
    valid = pd.read_csv('99-dataset/OutGen/OutGen_valid.csv')
    test = pd.read_csv('99-dataset/OutGen/OutGen_test.csv')
    train_input =  train['keywords']
    train_output = train['story']
    valid_input = valid['keywords']
    valid_output = valid['story']
    test_input = test['keywords']
    test_output = test['story']
    for name in ["train","valid","test"]:
        vocab = []
        for cat in ["input","output"]:
            df = eval(name+"_"+cat)
            if cat == 'input':
                word_len = df.apply(get_word_len_outline)
                sentence_len = df.apply(lambda x: len(x.split(',')))
            else:
                word_len = df.apply(get_word_len)
                sentence_len = df.apply(get_sentence_len)
            if cat == 'output':
                vocab+=df.apply(get_word_list).sum()
            print('word sentence len',name,cat,word_len.mean(),sentence_len.mean())
        print('vocab size',name,len(set(vocab)))
OutGen_stats()
