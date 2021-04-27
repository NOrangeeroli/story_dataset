import pandas as pd
deduplicate_story = pd.read_csv('32-deduplicate-story.csv')
df_endbleu = pd.read_csv('33-get-ending-bleu.csv')
list_highendbleu = df_endbleu[df_endbleu['ending_bleu']>0.8].story.tolist()
deduplicate_story['high_end_bleu'] = deduplicate_story.story.isin(list_highendbleu)
deduplicate_story.to_csv('34-filter-trivial-ending.csv')
import pdb;pdb.set_trace()
