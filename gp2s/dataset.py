import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from skeleton_data import p_cols, s_cols

# x: foot pressure, y: skel
class SkelDataset(Dataset):
    def __init__(self, train=True, csv_dir='skeleton_data', csv_file='walk_and_stop.csv', transform=None):
        
        print('read file {}'.format(csv_file))

        df = pd.read_csv(os.path.join(csv_dir,csv_file))

        self.x = torch.Tensor(df.iloc[:,p_cols].values)
        self.y = torch.Tensor(df.iloc[:,s_cols].values)
        
        pivot = int(len(self.y) * 0.8)

        if train:
            self.x = self.x[:pivot]
            self.y = self.y[:pivot]
        else:
            self.x = self.x[pivot:]
            self.y = self.y[pivot:]

        print(self.x.size())
        print(self.y.size())

        self.transform = transform
        
    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx], self.y[idx]


if __name__=='__main__':
    a = SkelDataset(train=True)
    b = SkelDataset(train=False)
