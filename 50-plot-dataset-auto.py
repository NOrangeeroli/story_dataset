from Rake_For_Chinese.src.Rake import Rake
import pandas as pd
from tools import start_debugger_on_exception
start_debugger_on_exception()
infile = '32-deduplicate-story.csv'
stories = pd.read_csv(infile)
obj = Rake()
stop_path = "Rake_For_Chinese/data/stoplist/中文停用词表(1208个).txt"
conj_path = "Rake_For_Chinese/data/stoplist/中文分隔词词库.txt"
obj.initializeFromPath(stop_path, conj_path)
def get_keywords(x):
    result = obj.extractKeywordFromString(x)
    result = [(x,y) for x,y in result if y >1]
    result = [x for x,y in result] 
    result = sorted(result, key=len)
    r = []
    for w in result:
        if len(r)>0:
            if w.startswith(r[-1]):
                continue
        r.append(w)
    return ','.join(r)
stories['keywords']= stories.apply(lambda x: get_keywords(x.story),axis = 1)
stories.to_csv('50-plot-dataset-auto.csv')
import pdb;pdb.set_trace()
