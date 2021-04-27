import nltk
import pandas as pd
# from tokenizer import SimpleTokenizer
from utils import split_by_fullstop
# tokenizer = SimpleTokenizer(method="nltk")
r = pd.read_csv('32-deduplicate-story.csv',encoding="utf_8_sig")
result_list = r.story.values.tolist()
result_list = [x.strip() for x in result_list]
endings = [split_by_fullstop(x)[-1] for x in result_list]

s_bleu = []
# for k in [1,2,3,4,5,6,7,8]:
#     s_bleu[k]=[]
for i,ending in enumerate(endings):
    print(i)
    refs = [result_list[j] for j in range(len(result_list)) if j != i]
    refs = [x for y in refs for x in split_by_fullstop(y)]
    references = [list(x) for x in refs]
    hypothesis = list(ending)
    bs = nltk.translate.bleu_score.sentence_bleu(references, hypothesis)
    # for k in [1,2,3,4,5,6,7,8]:
    s_bleu.append(bs)
    print('bs:',bs)
r['ending_bleu'] = s_bleu
r['ending'] = endings
r.to_csv('33-get-ending-bleu.csv',encoding="utf_8_sig")
import pdb;pdb.set_trace()
