import torch
from models.linear import LinearRegressor
from models.lstm import LSTMRegressor
from inference import orthotics
import os

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

class SkelInferer:
    def __init__(self, gpu_ids=0, model_dir="logs/", model_file='gp2s_b1024_e300_lr0_0001_mse.pt', fake=False):
        self.fake = fake
        if self.fake:
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
            with torch.no_grad():
                y = self.net(x)
            s = y.cpu().numpy()
            return s

class OrthoticsInferer:
    def __init__(self, gpu_ids=0, model_dir="logs/", model_file='orthotics_lstm_b1_e400_lr0_0002_mse.pt', seq_len=5000):
        print("load orthtics model")
        
        self.save_dir = "static/orthotics.csv"
        if 'skel_data' not in os.listdir():
            self.save_dir = os.path.join("..", self.save_dir)
        
        self.x_series = deque(maxlen=seq_len)
        self.seq_len = seq_len
        self.is_making = False
        self.pbar = tqdm(total=seq_len)
        print("orthtics model loaded")

    def make_orthotics(self):
        self.is_making = True
        gyro_data = np.array(self.x_series)
        gdf = pd.DataFrame(gyro_data)
        gdf.to_csv("sample_gyro_data.csv")

        self.x_series.clear()

        left, right = orthotics(gyro_data=gyro_data, model_type='lstm', model_file='orthotics_lstm_b1_e400_lr0_0002_mse.pt')
        left, right = left[0], right[0]

        mid = np.zeros((orthotic_height, orthotic_height - orthotic_width * 2), dtype=np.int32)
        
        print("l.s = {}, r.s = {}, m.s = {}".format(left.shape, right.shape, mid.shape))
        pair = np.hstack([left, mid, right])
        print("p.s = {}".format(pair.shape))

        pair = pair.astype(np.uint8)
        pair = cv2.resize(pair, dsize=(100, 100), interpolation=cv2.INTER_AREA)
        ks = 5
        rks = ks * 2 + 1
        pair = cv2.GaussianBlur(pair,(rks, rks),0)
        
        print("orthotics has been made!")
        print("save as {}".format(self.save_dir))
        df = pd.DataFrame(pair)
        df.to_csv(self.save_dir)
        self.is_making = False
        return pair
        # return left, right
    
    def feed(self, foot):
        # print("feed {}, is made = {}".format(foot, self.is_made))
        
        self.x_series.append(foot)
        self.pbar.update(1)
        
        if len(self.x_series) == self.seq_len and self.is_making:
            self.pbar.close()
            self.pbar = tqdm(total=self.seq_len)
            # Thread(target=self.make_orthotics).start()
            self.make_orthotics()
        
if __name__ == "__main__":

    df = pd.read_csv("skeleton_data/skeleton_walking.csv")
    s = df.iloc[:,1:45].values
    pivot = s.shape[0] // 2
    

    inferer = OrthoticsInferer()

    for i in np.r_[pivot: pivot + 6000]:
        inferer.feed(s[i])

    pair = inferer.make_orthotics()
    cv2.imshow("pair", pair)
    cv2.waitKey(0)


