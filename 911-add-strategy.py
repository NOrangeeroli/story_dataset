import pandas as pd

from tools import start_debugger_on_exception
start_debugger_on_exception()
data = pd.read_csv('91-adjust_bias.csv')
data['story'] = data.apply(lambda x: x['context'].replace('<MASK>',x['True']),axis = 1)

raw_data = pd.read_csv('70-read-annotated-storiepairs-from-raw.csv')
raw_data['story'] = raw_data.apply(lambda x: x['context'].replace('<MASK>',x['True']),axis = 1)
def get_strategy(x):
    story = x['story']
    if story in raw_data.story.tolist():
        r = raw_data.set_index('story').loc[story]['strategy']
        try:
            assert r in [1,2]
        except:
            r = 1
        return r

    else:
        k = raw_data[raw_data['story'].apply(lambda x: x.startswith(story[:10]))].story.values[0]
        for i in range(len(story)):
            if k[:i]!=story[:i]:
                print(k[:i])
                break
        import pdb;pdb.set_trace()

data['strategy'] = data.apply(lambda x: get_strategy(x),axis = 1)
data.to_csv('91-adjust_bias.csv')
import pdb;pdb.set_trace()
