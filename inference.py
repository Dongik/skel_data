import pandas as pd
import os
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import SkelDataset, OrthoticDataset
from models.linear import LinearRegressor
from models.lstm import LSTMRegressor

num_gyro = 44
num_skel = 51
orthotic_width = 10
orthotic_height = 30
seq_len = 5000

# Predict skeleton from gyro data
# gyro_data is Numpy array(data_num x 44)
# (optional) skel_data is also Numpy array(data_num x 51, for Calculating Loss)
# Two options for Loading Model
#   1. model: get model from memory
#   2. model_dir, model_file: Load model from file path
# Return predicted skeleton: Numpy array(data_num x 51)
def gp2s(gyro_data=None, skel_data=None, model=None, model_dir='logs/', model_file='gp2s.pt', batch_size=512, gpu_ids=0):

    test_dataset = SkelDataset(train=False, data_x=gyro_data, data_y=skel_data)
    
    test_loader = DataLoader(dataset=test_dataset,
         batch_size=batch_size,
         shuffle=False,
         num_workers=32)

    # Device
    device = torch.device('cuda:{}'.format(gpu_ids)) if torch.cuda.is_available() else torch.device('cpu') 
        
    # Calculate loss when skeleton data exist
    if skel_data is not None:
        criterion = nn.MSELoss()
  
    # Load Model
    if model is not None:
        net = model.to(device)
    else:
        net = LinearRegressor(num_gyro, num_skel)
        net.load_state_dict(torch.load(os.path.join(model_dir, model_file)))
        net = net.to(device)

    # Inference
    net.eval()
    with torch.no_grad():
        test_loss = 0
        test_len = 0
        test_pred = torch.empty(0,num_skel)
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
    
            pred = net(x)
            
            # Calc loss
            if skel_data is not None:
                loss = criterion(pred, y)
                test_loss += loss
                test_len += len(x)
            
            # Concat predictions
            test_pred = torch.cat([test_pred, pred.cpu()])
    
    # Print loss when skeleton data exist
    if skel_data is not None:
        print('test loss: {}'.format(test_loss / test_len))
    
    print(test_pred.size())
    return test_pred.numpy()

# Predict orthotics from gyro data
# gyro_data is Numpy array(subject_num, seq_num(must 5000) x 44)
# 여러개의 csv를 읽을 땐 dataset.py의 read_orthotics_data()를 사용하시면 되고,
# 만약 한개의 csv만 사용한다면, numpy로 변환만 한 후 함수호출하셔도 됩니다.
# (optional) orthotic_left(right) is also Numpy array(data_num x 30 x 10, for Calculating Loss)
# model_type: lstm or linear (must be matched with .pt file)
# Two options for Loading Model
#   1. model: get model from memory
#   2. model_dir, model_file: Load model from file path
# Return predicted orthotics(left, right): each Numpy array(data_num x 30 x 10)
def orthotics(gyro_data=None, orthotic_left=None, orthotic_right=None, model_type='linear', model=None, model_dir='logs/', model_file='orthotics_fc.pt', batch_size=1, gpu_ids=0):
    data_y = None
    # concat left and right orthotics
    if orthotic_left is not None and orthotics_right is not None:
        data_y = torch.concat([orthotic_left, orthotic_right], dim=1)
    
    # if data has no batch, unsqueeze data
    gyro_data = torch.Tensor(gyro_data)
    if len(gyro_data.size()) < 3:
        gyro_num = len(gyro_data)
        if gyro_num > seq_len:
            gyro_data = gyro_data[:seq_len]
            # repeat when data length < 5000
        else:
            gyro_data = torch.cat([gyro_data]*(seq_len//gyro_num + 1))[:seq_len]    
        gyro_data = torch.unsqueeze(gyro_data, dim=0)
    test_dataset = OrthoticDataset(train=False, data_x=gyro_data, data_y=data_y)
    
    test_loader = DataLoader(dataset=test_dataset,
         batch_size=batch_size,
         shuffle=False,
         num_workers=32)

    # Device
    device = torch.device('cuda:{}'.format(gpu_ids)) if torch.cuda.is_available() else torch.device('cpu') 
        
    # Calculate loss when skeleton data exist
    if data_y is not None:
        criterion = nn.MSELoss()
  
    # get model
    if model is not None:
        net = model.to(device)
    # load model
    else:
        if model_type == 'lstm':
            net = LSTMRegressor(num_gyro, orthotic_height * orthotic_width * 2)
        elif model_type == 'linear':
            net = LinearRegressor(num_gyro*seq_len, orthotic_height * orthotic_width * 2)
        # load model file
        net.load_state_dict(torch.load(os.path.join(model_dir, model_file)))
        net = net.to(device)

    # Inference
    net.eval()
    with torch.no_grad():
        test_loss = 0
        test_len = 0
        test_pred = torch.empty(0,orthotic_height*orthotic_width*2)
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
         
            if model_type == 'linear':
                x= x.reshape(x.size(0), x.size(1)*x.size(2))
                pred = net(x)
            elif model_type == 'lstm':
                hc = net.init_hidden_cell(batch_size)
                pred, hc = net(x, hc)
            
            # Calc loss
            if data_y is not None:
                loss = criterion(pred, y)
                test_loss += loss
                test_len += len(x)
            
            # Concat predictions
            test_pred = torch.cat([test_pred, pred.cpu()])
    
    # Print loss when skeleton data exist
    if data_y is not None:
        print('test loss: {}'.format(test_loss / test_len))
    
    # Smooth and Reshape to left and right orthotics
    test_pred = ((test_pred - test_pred.min()) / test_pred.max() * 256).type(torch.int32)
    test_pred[test_pred < 50] = 0

    test_pred = test_pred.view(-1, orthotic_height*2, orthotic_width)

    left = test_pred[:, :orthotic_height]
    right = test_pred[:, orthotic_height:]

    return left.numpy(), right.numpy()

def test_infer_orthotics():
    data = OrthoticDataset(train=True,train_ratio=1.0)

    left, right = orthotics(gyro_data=data.x, model_type='linear', model_file='orthotics_fc.pt', batch_size=5)

    fig = plt.figure() # rows*cols 행렬의 i번째 subplot 생성
    rows = len(data.x)
    cols = 2
    i = 1
 
    for i in range(1, 11, 2):
        # imshow left
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(left[i//2])
        ax.set_xlabel(str(i//2))
        ax.set_xticks([]), ax.set_yticks([])
        # imshow right
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(right[i//2])
        ax.set_xlabel(str(i//2))
        ax.set_xticks([]), ax.set_yticks([])
    plt.show()  


if __name__=="__main__":
    import numpy as np
    
    print('GP2Skel')
    data = np.zeros((100, 44))
    pred = gp2s(gyro_data=data, model_file='gp2s.pt')
    
    print('Shape of X:', data.shape)
    print('Shape of Pred:', pred.shape)
    print()

    print('Orthotics')
    test_infer_orthotics()
    '''
    data = np.ones((1, 5000, 44))
    left, right = orthotics(gyro_data=data, model_type='linear', model_file='orthotics_fc.pt')
    
    print('Shape of X:', data.shape)
    print('Left:')
    print(left)
    print('Right:')
    print(right)
    '''
