import pandas as pd
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import SkelDataset
from models.linear import LinearRegressor

num_gyro = 44
num_skel = 51

# Predict skeleton from gyro data
# gyro_data is Numpy array(data_num x 44)
# (optional) skel_data is also Numpy array(data_num x 51)
# Return predicted skeleton: Numpy array(data_num x 51)
def gp2s(gyro_data=None, skel_data=None, model_dir='./', model_file='model.pt', batch_size=512, gpu_ids=0):

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
   
if __name__=="__main__":
    import numpy as np

    data = np.zeros((100, 44))
    pred = gp2s(gyro_data=data)
    
    print('Shape of X:', data.shape)
    print('Shape of Pred:', pred.shape)
