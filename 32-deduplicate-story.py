import pandas as pd
import glob
raw_stories = pd.read_csv('30-raw_stories.csv')
manual_csvs =glob.glob('31-drop-duplicates*manual.csv')
duplicated= pd.concat([pd.read_csv(c,encoding="utf_8_sig") for c in manual_csvs])
duplicated_entries = duplicated[duplicated['Unnamed: 0']==3000]['0'].tolist()

stories = raw_stories[~raw_stories.story.isin(duplicated_entries)]
stories.to_csv('32-deduplicate-story.csv',encoding="utf_8_sig")
