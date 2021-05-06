import torch
from skel_data.models.linear import LinearRegressor
from skel_data.models.lstm import LSTMRegressor
import os

from collections import deque


num_gyro=44
num_skel=51
orthotic_width = 10
orthotic_height = 30

class SkelInferer:
    def __init__(self, gpu_ids=0, model_dir="logs/", model_file='gp2s_b1024_e300_lr0_0001_mse.pt'):
        device = torch.device('cuda:{}'.format(gpu_ids)) if torch.cuda.is_available() else torch.device('cpu') 
        net = LinearRegressor(num_gyro, num_skel)
        net.load_state_dict(torch.load(os.path.join(model_dir, model_file)))
        self.net = net.to(device)
        self.net.eval()
        self.device = device
    
    def to_skel(self, gp):
    
        x = torch.Tensor(gp)
        x = x.to(self.device)
        with torch.no_grad():
            y = self.net(x)
        s = y.cpu().numpy()
        return s

class OrthoticsInferer:
    def __init__(self, gpu_ids=0, model_dir="logs/", model_file='orthotics_fc.pt', seq_len=5000):
        device = torch.device('cuda:{}'.format(gpu_ids)) if torch.cuda.is_available() else torch.device('cpu') 
        net = LinearRegressor(num_gyro*seq_len, orthotic_height * orthotic_width * 2)
        net.load_state_dict(torch.load(os.path.join(model_dir, model_file)))
        self.net = net.to(device)
        self.net.eval()
        self.device = device
        self.x_series = deque(maxlen=seq_len)
        self.seq_len = seq_len

    def get_orthotics(self):
        if len(self.x_series) == self.seq_len:
            x = torch.Tensor(np.array(self.x_series))
            x = torch.unsqueeze(x, dim=0)
            x = x.to(self.device)
            x = x.reshape(x.size(0), x.size(1)*x.size(2))

            test_pred = torch.empty(0,orthotic_height*orthotic_width*2)

            with torch.no_grad(): 
                pred = self.net(x)
                test_pred = torch.cat([test_pred, pred.cpu()])

            test_pred = ((test_pred - test_pred.min()) / test_pred.max() * 256).type(torch.int32)
            test_pred[test_pred < 50] = 0

            test_pred = test_pred.view(-1, orthotic_height*2, orthotic_width)

            left = test_pred[:, :orthotic_height]
            right = test_pred[:, orthotic_height:]

            print("ls = {}, rs = {}".format(left.shape, right.shape))
            # self.x_series.clear()
            return left, right
        else:
            print("not enough dataset")

    def feed(self, gp):
        self.x_series.append(gp)

        return self.get_orthotics()