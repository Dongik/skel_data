import os
import pandas as pd

import torch
from torch.utils.data import Dataset

# x: foot pressure, y: skel
class SkelDataset(Dataset):
    def __init__(self, train=True, csv_dir='legacy_skeleton_data', csv_file='keep_walk.csv', transform=None):
        print('read file {}'.format(csv_file))
        df = pd.read_csv(os.path.join(csv_dir,csv_file))
        self.x = torch.Tensor(df.iloc[:,pd.np.r_[3:9, 26:42, 44:66]].values)
        self.y = torch.Tensor(df.iloc[:, 84:186].values)
        
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
