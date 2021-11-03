


from nn.torch_models import LSTM
import torch
from models.linear import LinearRegressor
from models.lstm import LSTMRegressor
from inference import orthotics
import os
from torch import nn

import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm


from threading import Thread
import cv2

num_gyro=44
num_skel=51
orthotic_width = 10
orthotic_height = 30
import time

class Inference:
    def __init__(self, infer_type, model=None, model_type='lstm', model_dir='logs/', model_file='gp2s.pt', batch_size=512, n_layers=8, seq_len=32, loss='mse', gpu_ids=0):
        self.infer_type = infer_type
        self.model_type = model_type
        self.model_dir = model_dir
        self.model_file = model_file
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.loss = loss
        self.flag = False

        self.data_seq = torch.zeros((32, 44))
        
        # Device
        self.device = torch.device('cuda:{}'.format(gpu_ids)) if torch.cuda.is_available() else torch.device('cpu') 
        self.num_workers = torch.get_num_threads()

        # Loss
        self.criterion = nn.MSELoss()
        # Empty Dataset
        self.dataset = None
        self.extra_data = None
        self.dataloader = None
        # Load Model
        if model is not None:
            self.net = model.to(self.device)
        else:
            if self.model_type == 'linear':
                self.net = LinearRegressor(num_gyro, num_skel)
            elif self.model_type == 'lstm':
                self.net = LSTMRegressor(num_gyro, num_skel, num_layers=self.n_layers)
            self.net.load_state_dict(torch.load(os.path.join(model_dir, model_file)))
            self.net = self.net.to(self.device)
            self.net.eval()

    def setAttr(self, batch_size=None, seq_len=None, gpu_ids=None, loss='None'):
        if batch_size is not None:
            self.batch_size = batch_size
        if seq_len is not None:
            self.seq_len = seq_len
        if gpu_ids is not None:
            self.gpu_ids = gpu_ids
            self.device = torch.device('cuda:{}'.format(gpu_ids)) if torch.cuda.is_available() else torch.device('cpu') 

    def infer(self, gyro_data):
        if not self.flag:
            self.flag = True
            if not torch.is_tensor(gyro_data):
                gyro_data = torch.Tensor(gyro_data)

            self.data_seq = torch.cat([self.data_seq, gyro_data])[-self.seq_len:,:]
            
            x = torch.unsqueeze(self.data_seq, 0)
            with torch.no_grad():

                batch_size = 1
                x = x.to(self.device)
                    
                if self.model_type == 'linear':
                    pred = self.net(x)
                elif self.model_type == 'lstm':
                    hc = self.net.init_hidden_cell(batch_size)
                    pred, hc = self.net(x, hc)
                    print("pred.shape = {}".format(pred.shape))

                pred = pred.cpu()
        
                self.flag = False
                return pred.numpy()
        else:
            return None
class SkelInferer:
    def __init__(self, gpu_ids=0, model_dir="logs/", model_file='gp2s_lstm_e500_n8_b1024_lr0_0001_seq32_str5_mse.pt', fake=False):
        self.fake = fake
        if self.fake:
            print("use fake data")
            self.start_time = time.time()
            self.count = 0
            df = pd.read_csv("skel_data/skeleton_data/skeleton_walking.csv")
            self.skel = df.iloc[:,45:].values
            self.total = self.skel.shape[0]

        else:
            if 'skel_data' in os.listdir():
                model_dir = os.path.join('skel_data', model_dir)

            print("load skel model")
            device = torch.device('cuda:{}'.format(gpu_ids)) if torch.cuda.is_available() else torch.device('cpu') 

            if 'lstm' in model_file:
                num_layers = int(model_file.split("_")[3][1:])
                net = LSTMRegressor(num_gyro, num_skel, num_layers=num_layers)
            else:
                net = LinearRegressor(num_gyro, num_skel)

            net.load_state_dict(torch.load(os.path.join(model_dir, model_file)))
            self.net = net.to(device)
            self.net.eval()
            self.device = device
            print("skel model loaded")
    
    def to_skel(self, gp):
        if self.fake:
            dt = time.time() - self.start_time
            fn = int(dt * 30) % self.total
            return self.skel[fn]
        else:
            x = torch.Tensor(gp)
            x = x.to(self.device)
            x = torch.unsqueeze(x, 0)
            with torch.no_grad():
                y = self.net(x)
            s = y.cpu().numpy()
            return s

def make_orthotics(data, base_dir="", model_file='orthotics_lstm_b1_e400_lr0_0002_mse.pt', save_dir="static/orthotics.csv"):
    left, right = orthotics(gyro_data=data, model_type='lstm', model_file=model_file)
    left, right = left[0], right[0]

    mask = cv2.imread(os.path.join(base_dir, "mask.png"))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("mask", mask)

    mid = np.zeros((orthotic_height, orthotic_height - orthotic_width * 2), dtype=np.int32)

    print("l.s = {}, r.s = {}, m.s = {}".format(left.shape, right.shape, mid.shape))
    pair = np.hstack([left, mid, right])
    pair = pair.astype(np.uint8)
    
    r_max = np.amax(pair, axis=1)
    print(r_max)

    print("p.s = {}".format(pair.shape))
    for i in range(pair.shape[0]):
        for j in range(pair.shape[1]):
            if pair[i][j] < 50:
                pair[i][j] = r_max[i]
            
    # pair = pair.astype(np.uint8)

    # cv2.imshow("pair", pair)
    # print(pair)
    # print("p.s = {}".format(pair.shape))

    # pair = pair.astype(np.uint8)
    pair = cv2.resize(pair, dsize=(300, 300), interpolation=cv2.INTER_AREA)


    ks = 15
    rks = ks * 2 + 1
    pair = cv2.GaussianBlur(pair,(rks, rks),0)
    
    pair = cv2.bitwise_and(pair, mask)

    padding = np.zeros((330, 330))
    padding[:]

    pair = cv2.resize(pair, (100, 100))

    # cv2.imshow("pair", pair)
    # cv2.waitKey(0)

    print("orthotics has been made!")
    print("save as {}".format(save_dir))
    
    df = pd.DataFrame(pair)
    df.to_csv(save_dir)
    
    return pair



class OrthoticsInferer:
    def __init__(self, seq_len=5000):
        print("load orthtics model")
        
        self.save_dir = "static/orthotics.csv"
        if 'skel_data' not in os.listdir():
            self.save_dir = os.path.join("..", self.save_dir)
        
        self.x_series = deque(maxlen=seq_len)
        self.seq_len = seq_len
        self.is_making = False
        # self.pbar = tqdm(total=seq_len)
        print("orthtics model loaded")

    def make_orthotics(self):
        self.is_making = True
        gyro_data = np.array(self.x_series)
        self.x_series.clear()
        pair = make_orthotics(gyro_data,base_dir="skel_data", save_dir=self.save_dir)
        self.is_making = False
        return pair
        # return left, right
    
    def feed(self, foot):
        # print("feed {}, is made = {}".format(foot, self.is_made))
        
        self.x_series.append(foot)
        print("feed skel_data {}/{}".format(len(self.x_series),self.seq_len))
        
        if len(self.x_series) == self.seq_len and self.is_making:
            self.make_orthotics()
        
if __name__ == "__main__":

    df = pd.read_csv("sample_gyro_data.csv")
    print("s.s = {}".format(df.shape))
    s = df.iloc[:,1:45].values
    print("s.s = {}".format(s.shape))
    
    make_orthotics(s, save_dir="../static/orthotics.csv")

    
    # pivot = s.shape[0] // 2
    

    # inferer = OrthoticsInferer()
    

    # for i in np.r_[:5000]:
    #     inferer.feed(s[i])

    # pair = inferer.make_orthotics()
    
    
    
    # cv2.imshow("pair", pair)
    # cv2.waitKey(0)


