import os
import pandas as pd
from torch.utils.data import Dataset

class DataSetBert(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file).reset_index()
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = self.data.loc[idx,'source']
        target = self.data.loc[idx,'target']
        sample = {"source": source, "target": target}
        return sample
