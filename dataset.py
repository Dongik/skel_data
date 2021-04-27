import os
import pandas as pd

import torch
from torch.utils.data import Dataset

# x: foot pressure, y: skel
class SkelDataset(Dataset):
    def __init__(self, train=True, data_x = None, data_y = None, csv_dir='skeleton_data', csv_file='keep_walk.csv', transform=None):
        # data array exist
        if data_x is not None:
            self.x = torch.Tensor(data_x)
            if data_y is not None:
                self.y = torch.Tensor(data_y)
            else:
                self.y = torch.zeros(len(data_x), 1)
        
        # Read csv file
        else:
            print('read file {}'.format(csv_file))
            df = pd.read_csv(os.path.join(csv_dir,csv_file), index_col=0)
            self.x = torch.Tensor(df.iloc[:,:44].values)
            self.y = torch.Tensor(df.iloc[:,44:].values)
        
            # Split dataset if Train
            pivot = int(len(self.y) * 0.8)
            if train:
                self.x = self.x[:pivot]
                self.y = self.y[:pivot]
            else:
                self.x = self.x[pivot:]
                self.y = self.y[pivot:]

        print("X:", self.x.size())
        print("Y:", self.y.size())

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
