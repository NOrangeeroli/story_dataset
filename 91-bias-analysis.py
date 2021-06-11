import pandas as pd
from scipy import stats
from snownlp import SnowNLP
from nltk.translate.bleu_score import sentence_bleu
import jieba
from bert_score import BERTScorer
from tools import start_debugger_on_exception
start_debugger_on_exception()
def read_data(f = '70-read-annotated-storiepairs-from-raw.csv'):
    return pd.read_csv(f)
def cut_jieba(x):
    return list(jieba.cut(x, cut_all=False))
def length_bias():
    df = read_data()
    sample1 = df['True'].apply(lambda x: len(cut_jieba(x)))
    sample2 = df['False'].apply(lambda x: len(cut_jieba(x)))
    df['s_t'] = sample1
    df['s_f'] = sample2
    df['diff'] = df['s_f']-df['s_t']
    print('True',sample1.mean(),'False',sample2.mean())
    return stats.ttest_ind(sample1,sample2, equal_var = False)[1]
def sentiment_bias():
    df = read_data()
    sample1 = df['True'].apply(lambda x: SnowNLP(x).sentiments)
    sample2 = df['False'].apply(lambda x: SnowNLP(x).sentiments)
    df['s_t'] = sample1
    df['s_f'] = sample2
    #import pdb;pdb.set_trace()
    print('True',sample1.mean(),'False',sample2.mean())
    return stats.ttest_ind(sample1,sample2, equal_var = False)[1]
    
def bleu_bias(n = 1):
    df = read_data()
    sample1 = df.apply(lambda x: sentence_bleu([cut_jieba(x.context)], cut_jieba(x['True']),weights=[1/float(n)]*n),axis = 1)
    sample2 = df.apply(lambda x: sentence_bleu([cut_jieba(x.context)], cut_jieba(x['False']),weights = [1/float(n)]*n),axis = 1)
    #import pdb;pdb.set_trace()
    print('True',sample1.mean(),'False',sample2.mean())
    return stats.ttest_ind(sample1,sample2, equal_var = False)[1]

def negetion_bias():
    df = read_data()
    sample1 = df['True'].apply(lambda x: ('没' in x) or ('不' in x))
    sample2 = df['False'].apply(lambda x: ('没' in x) or ('不' in x))
    #import pdb;pdb.set_trace()
    print('True',sample1.mean(),'False',sample2.mean())
    return stats.ttest_ind(sample1.apply(int),sample2.apply(int), equal_var = False)[1]

def bertscore_bias():
    scorer = BERTScorer(lang="zh", rescale_with_baseline=True)
    df = read_data()
    sample1 = df.apply(lambda x: scorer.score([x.context],[x['True']])[2].item(),axis = 1)
    sample2 = df.apply(lambda x: scorer.score([x.context],[x['False']])[2].item(),axis = 1)
    print('True',sample1.mean(),'False',sample2.mean())
    return stats.ttest_ind(sample1,sample2, equal_var = False)[1]

print(length_bias())
print(sentiment_bias())
print(bleu_bias())
print(negetion_bias())
#print(bertscore_bias())
import pdb;pdb.set_trace()
