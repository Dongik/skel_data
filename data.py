
from os.path import join
import glob
from torch.utils.data import Dataset
from torch import optim
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

root_dir = "/data/balance/walk"

def read_bal_data(root_dir="")


class BalSeries(Dataset):
    def __init__(self, dataset_range=(0, 1), num_frames=81, stride=1, num_workers=4, root_dir="/data/balance/walk"):
        self.range = dataset_range
        self.num_workers = num_workers
        
        p_sets = []
        g_sets = []
        s_sets = []

        p_cols = list(range(9, 25)) + list(range(50, 66))
        g_cols = list(range(3, 10)) + list(range(44, 51))
        s_cols = []
        for i in range(17):
            ii = 85 + i * 6
            for j in range(3):
                s_cols.append(ii + j)

        for csv_dir in tqdm(glob.glob(join(root_dir, "*/skeleton.csv"), recursive=True)):
            df = pd.read_csv(csv_dir)
            # print("df.shape = {}".format(df.shape))
            p_record = df.iloc[:, p_cols].values.astype(np.float32)
            g_record = df.iloc[:, g_cols].values.astype(np.float32)
            s_record = df.iloc[:, s_cols].values.astype(np.float32)
            for i in range(num_frames, len(p_record)):
                si = i - num_frames
                p_sets.append(p_record[si: i: stride])
                g_sets.append(g_record[si: i: stride])
                s_sets.append(s_record[si: i: stride])
        
        self.p_sets = np.array(p_sets)
        self.g_sets = np.array(g_sets)
        self.s_sets = np.array(s_sets)
       

    def __len__(self):
        return len(self.p_sets)

    def __getitem__(self, index):
        p = torch.tensor(self.p_sets[index])
        g = torch.tensor(self.g_sets[index])
        s = torch.tensor(self.s_sets[index])
        return p, g, s
