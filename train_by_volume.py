import csv
import pandas as pd
import torch
import numpy as np
import os
from os.path import isfile, isdir, getsize
from os import rename, listdir, mkdir

from shutil import copyfile
from numpy import pi, exp, sqrt

from tqdm import trange
from torch.utils.data import Dataset
from config import base_dir, skeleton_filename
from tqdm import tqdm
import pickle
from collections import deque
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from converter import to_plane
from cv2 import cv2

from itertools import cycle, islice

from multiprocessing import Pool

from multiprocessing.pool import ThreadPool

from torch.multiprocessing import Pool, Process, set_start_method

import glob

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

def normalize_skel(s_row):
    joints = []
    for i in range(17):
        j = i * 3
        joint = s_row[j], s_row[j + 1], s_row[j + 2]
        joints.append(joint)


def upsample_p(volume,row, size=(64, 64)):
    w, h = size
    ks = 29
    frame = np.zeros(self.size)
    amp = 1    
    for j in range(16):
        u, v = p_pos[j]
        lp = (row[7 + j] + 1) * amp
        rp = (row[29 + j] + 1) * amp
        
        frame[int((0.5 - u) * w)][int(v * h)] = lp 
        frame[int((0.5 + u) * w)][int(v * h)] = rp
    
    frame = cv2.GaussianBlur(frame,(ks, ks),0)
    # frame = frame.astype('uint8')
    return frame

def get_g_row(row):
    temprow = []
    temprow.extend(row[1:7])
    temprow.extend(row[23:29])
    return temprow




class SkelCsvReader:
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        df = pd.read_csv(csv_dir)
        self.df = df

    def p_dataset(self):
        lgs = df.columns.get_loc("left.ang.x")
        lgs = df.columns.get_loc("right.ang.x")
        return self.df.loc[:, pd.np.r_[:7, 26:42]]

    def g_dataset(self):
        lgs = df.columns.get_loc("left.ang.x")
        lgs = df.columns.get_loc("left.ang.x")
        return self.df.loc[:, pd.np.r_[3:9, 26:42]]
    
    def s_dataset(self):
        lgs = df.columns.get_loc("left.ang.x")
        lgs = df.columns.get_loc("left.ang.x")
        return self.df.loc[:, pd.np.r_[3:9, 26:42]]

    

class VolumeDatasetLoader:
    def __init__(self, depth=64, width=64, pivot=0.8, max_records=5, num_workers=12, normalize_skel=True):
        self.depth = depth
        self.pivot = pivot
        self.train_r = train_r
        self.max_records = max_records
        self.size = width, width
        self.norimalize_skel = normalize_skel
        self.num_workers = num_workers

        p_sets = []
        g_sets = []
        s_sets = []

        print("load data")

        for csv_dir in glob.glob("skeleton_data/*.csv"):
            df = pd.read_csv(csv_dir)
        
            lps = df.columns.get_loc("left.forces.0")
            rps = df.columns.get_loc("right.forces.0")
            pn = 16

            lgs = df.columns.get_loc("left.ang.x")
            rgs = df.columns.get_loc("right.ang.x")
            gn = 6
            
            p_record = df.iloc[:,pd.np.r_[lps:lps + pn, rps:rps + pn]].values
            g_record = df.iloc[:,pd.np.r_[lgs:lgs + gn, rgs:rgs + gn]].values

            v_shape = len(df.index), width, width

            volume = np.zeros(v_shape)
            
            for i, row in df.iterrows():
                p_upsampled = upsample_p(row, size=self.size)



class VolumeDataset(Dataset):
    def __init__(self, depth=64, width=64, train=True, train_r=0.8, max_records=5, num_workers=4, normalize_skel=True):
        self.depth = depth
        self.train = train
        self.train_r = train_r
        self.max_records = max_records
        self.size = width, width
        self.norimalize_skel = normalize_skel
        self.num_workers = num_workers
        self.load_datasets()

    def load_datasets(self):
        skel_dirs = self.read_skel_dirs()

        total_ct = 0
        
        print("calculating data length")

        for skel_dir in skel_dirs:
            df = pd.read_csv(skel_dir)
            total_ct += len(df.index)
        self.pbar = tqdm(total=total_ct)

        self.datasets = []
        self.p_sets = []
        self.g_sets = []
        self.s_sets = []

        print("load dataset")
        
        for skel_dir in self.read_skel_dirs():
            self.load_record(skel_dir)
        # # # if     

        self.pbar.close()


    def read_skel_dirs(self):
        skel_dirs = []

        skeleton_filename = "skeleton_v2.csv"
        for csv_dir in 
        for folder_name in os.listdir(base_dir):
            skel_dir = "{}/{}/{}".format(base_dir, folder_name, skeleton_filename)
            if isfile(skel_dir):
                skel_dirs.append(skel_dir)
                
        if self.max_records < len(skel_dirs):
            skel_dirs = skel_dirs[:self.max_records]

        div = int(len(skel_dirs) * self.train_r)

        if self.train:
            skel_dirs = skel_dirs[:div]
        else:
            skel_dirs = skel_dirs[div:]
        
        return skel_dirs
    
    def upsample_p(self, row):
        w, h = self.size
        ks = 29
        frame = np.zeros(self.size)
        amp = 1    
        for j in range(16):
            u, v = p_pos[j]
            lp = (row[7 + j] + 1) * amp
            rp = (row[29 + j] + 1) * amp
            
            frame[int((0.5 - u) * w)][int(v * h)] = lp 
            frame[int((0.5 + u) * w)][int(v * h)] = rp
        
        frame = cv2.GaussianBlur(frame,(ks, ks),0)
        # frame = frame.astype('uint8')
        return frame

    def get_g_row(self, row):
        temprow = []
        temprow.extend(row[1:7])
        temprow.extend(row[23:29])
        return temprow

    def load_record(self, file_dir):
        # if not isfile(file_dir):
        #     return None
        df = pd.read_csv(file_dir, sep=',')

        p_record = []
        g_record = []


        for i, row in df.iterrows():  # each row is a list
            p_upsampled = self.upsample_p(row)
            g_row = self.get_g_row(row)
            
            p_record.append(p_upsampled)
            g_record.append(g_row)
            # if self.depth < i:
            #     self.pbar.update(0.5)
        


        p_record = np.array(p_record, dtype=np.float32)
        g_record = np.array(g_record, dtype=np.float32)
            
        
        for i in range(self.depth, len(df.index)):

        # for i, row in df.iterrows():
            if self.depth < i:
                data = BalData()
                s, e = i - self.depth, i
                
                data.p = p_record[s: e]
                data.g = g_record[s: e]

                s_row = row[45:]
                data.s = np.array(s_row, dtype=np.float32)
                self.datasets.append(data)
                
                self.pbar.update(1)



    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        data:BalData = self.datasets[index]

        p = torch.tensor(data.p)
        g = torch.tensor(data.g)
        s = torch.tensor(data.s)
        return p, g, s


class VolumeNet(nn.Module):

    def _conv_layer_set(self, in_c, out_c):
        conv = nn.Sequential(
            nn.Conv3d(in_c,out_c, 3, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
        )
        # conv = nn.DataParallel(conv)
        return conv

    def __init__(self):
        super().__init__()
        
        self.input_width = 64
        width = self.input_width
        output_width = 3 * 17
    
        
        # self.gauss = GaussianBlur2d((9, 9), (1.5, 1.5))

        # self.conv1 = self._conv_layer_set(40,64)
        self.conv2 = self._conv_layer_set(1,64)
        self.conv3 = self._conv_layer_set(64,128)
        # self.conv4 = self._conv_layer_set(128,128)
        self.linear1 = nn.Sequential(
            # nn.Linear(256, 128),
            nn.Linear(128, output_width)
        )
        # self.linear1 = nn.DataParallel(self.linear1)
        

        self.input_vector2 = 12
        self.output_vector2 = 128
        self.num_layers2 = 1
        
        
        self.lstm2 = nn.LSTM(input_size=self.input_vector2, hidden_size=self.output_vector2, num_layers=self.num_layers2, batch_first=True)
        
        self.linear2 = nn.Sequential(
            nn.Linear(self.output_vector2, 64),
            nn.Linear(64, output_width)
        )
        # self.linear2 = nn.DataParallel(self.linear2)

        self.linear3 = nn.Sequential(
            nn.Linear(output_width * 2, 64),
            nn.Linear(64, output_width)
        )
        # self.linear3 = nn.DataParallel(self.linear3)


    
    def forward(self, p_v, g_series):
        # output = self.conv1(p_volume)
        # p_v = self.gauss(p_volume)
        p_v = torch.unsqueeze(p_v, 1)
        output = self.conv2(p_v)
        output = self.conv3(output)
        
        output = output[:,:,-1,-1,-1]
        
        result_pressure = self.linear1(output)
        
        self.lstm2.flatten_parameters()
        output, _ = self.lstm2(g_series) #(hidden, cell) 데이터는 사용하지 않음
        output = output[:,-1,:]
        result_gyro = self.linear2(output)
   
        concat = torch.cat([result_gyro,result_pressure],dim=1) 
        
        return self.linear3(concat)
    


p_pos = [
    (0.267, 0.832), # 0
    (0.181, 0.880), # 1
    (0.131, 0.886), # 2
    (0.066, 0.881), # 3
    (0.292, 0.655), # 4
    (0.209, 0.680), # 5
    (0.133, 0.689), # 6
    (0.058, 0.705), # 7
    (0.279, 0.554), # 8
    (0.255, 0.389), # 12
    (0.185, 0.438), # 13
    (0.118, 0.463), # 14
    (0.217, 0.242), # 17
    (0.119, 0.251), # 18
    (0.194, 0.079), # 21
    (0.107, 0.094), # 22
]

if __name__ == "__main__":
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda")


    model = VolumeNet()
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    # loss & optimizer setting
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters())
    # load data

    DEBUG = False


    max_records = 100
    
    num_workers = 6
    batch_size = 2**8
    
    train_ratio = 0.8

    if max_records == 1:
        train_ratio = 1

    epoch = 100

    if DEBUG:
        max_records = 5
        epoch = 1


    train_dataset = BalDataset(train=True, max_records=max_records, train_r=train_ratio, num_workers=num_workers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # start training
    with tqdm(total=epoch) as e_pbar:
        for i in range(epoch):
            batch_loss = 0.0
            # pbar = tqdm(total=len(train_loader))
            for bi, (p, g, s) in enumerate(tqdm(train_loader, leave=False)):
                p = p.to(device)
                g = g.to(device)
                s = s.to(device)

                model.train()
                outputs = model(p, g)
                loss = criterion(outputs, s)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()

            if i % 25 == 0:
                print('Epoch {}, Loss {:.5f}'.format(i, batch_loss))
            e_pbar.update(1)
            pf = {
                "Epoch" : "{}/{}".format(i, epoch),
                "Loss" : "{:.5f}".format(batch_loss)
            }
            e_pbar.set_postfix(pf)


    # model save
    PATH = './model_by_volume.pth'
    torch.save(model.state_dict(), PATH)

    # start eval
    model.eval()

    test_dataset = BalDataset(train=False, max_records=max_records, train_r=train_ratio)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    with torch.no_grad():
        eval_loss =0.0
        for  i, (p, g, s) in enumerate(tqdm(test_loader, leave=False)):
            p = p.to(device)
            g = g.to(device)
            s = s.to(device)

            predict = model(p, g)
            loss = criterion(predict, s)
            eval_loss += loss.item()
        print('final loss : {}'.format(eval_loss))
        

    # save result
    # save_result(finalpredict,"newnewY.csv")
    # save_result(testy.cpu().numpy(), "testy.csv")

