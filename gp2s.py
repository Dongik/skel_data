import pandas as pd
import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
import torch.nn.functional as F

from dataset import SkelDataset
from models.linear import LinearRegressor

from inference import gp2s

num_gyro = 44
num_skel = 51

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float, metavar='NAME', help='Learning Rate')
    parser.add_argument('-e', '--epochs', default=200, type=int, metavar='NAME', help='Number of Epochs')
    parser.add_argument('-b', '--batch_size', default=512, type=int, metavar='NAME', help='Batch Size')
    parser.add_argument('-l', '--loss', default='mse', type=str, metavar='NAME', help='Type of Loss Function')
    parser.add_argument('-gpu', '--gpu_ids', default=0, type=int, metavar='NAME', help='GPU Numbers')
    args = parser.parse_args()

    #tr = torch.nn.Sequential(
    #    torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
    tr = None


    # DATASETS
    #data_name = 'skeleton_stop_walk_repeat.csv'    
    train_dataset = SkelDataset(train=True, transform=tr)
    test_dataset = SkelDataset(train=False, transform=tr)
    
    train_loader = DataLoader(dataset=train_dataset,
         batch_size=args.batch_size,
         shuffle=True,
         num_workers=32)

    # Training Parameters
    device = torch.device('cuda:{}'.format(args.gpu_ids)) if torch.cuda.is_available() else torch.device('cpu') 
    lr = args.learning_rate
    epochs = args.epochs
    
    # Models
    net = LinearRegressor(num_gyro, num_skel).to(device)
    
    args.loss = args.loss.lower()
    if args.loss == 'rmse':
        criterion = nn.RMSELoss() 
    # mse
    else :
        criterion = nn.MSELoss()

    #optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    
    net.train()
    
    # train
    for epoch in range(epochs):
        train_loss = 0
        train_len = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
    
            pred = net(x)
            loss = criterion(pred, y)
    
            loss.backward()
            optim.step()
            optim.zero_grad()
    
            train_loss += loss
            train_len += len(x)
    
        print('epoch: {}'.format(epoch+1))
        print('tr_loss: {}'.format(train_loss.item() / train_len))
   

    print('Save Model ...')
    model_dir = 'logs'
    model_file = 'gp2s.pt'
    model_path = os.path.join(model_dir, model_file)
    torch.save(net.state_dict(), model_path)
    print(model_path, 'Saved.')
    

    # Test (Prediction)
    test_pred = gp2s(gyro_data=test_dataset.x, skel_data=test_dataset.y, model_dir=model_dir, 
            model_file=model_file, batch_size=args.batch_size, gpu_ids=args.gpu_ids)
    test_pred = torch.Tensor(test_pred)

    # Save Logs

    # Save csv
    '''
    test_pred = torch.cat([torch.zeros(len(test_pred),84), test_pred], dim=1)
    test_y = torch.cat([torch.zeros(len(test_dataset.y),84), test_dataset.y], dim=1)
    
    df_pred = pd.DataFrame(test_pred.numpy())
    df_y = pd.DataFrame(test_y.numpy())
    
    
    print('Save CSVs ...')
    df_pred.to_csv('skel_pred.csv', sep=',', index=False)
    df_y.to_csv('skel_y.csv', sep=',', index=False)
    ''' 
