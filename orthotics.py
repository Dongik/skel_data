import pandas as pd
import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
import torch.nn.functional as F

from dataset import OrthoticDataset
from models.linear import LinearRegressor
from models.lstm import LSTMRegressor

from inference import orthotics

num_gyro = 44
orthotic_height = 30
orthotic_width = 10
seq_len = 5000

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, metavar='NAME', help='Learning Rate')
    parser.add_argument('-e', '--epochs', default=500, type=int, metavar='NAME', help='Number of Epochs')
    parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='NAME', help='Batch Size')
    parser.add_argument('-l', '--loss', default='mse', type=str, metavar='NAME', help='Type of Loss Function')
    parser.add_argument('-m', '--model', default='linear', type=str, metavar='NAME', help='Type of Model')
    parser.add_argument('-gpu', '--gpu_ids', default=0, type=int, metavar='NAME', help='GPU Numbers')
    args = parser.parse_args()

    #tr = torch.nn.Sequential(
    #    torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
    tr = None

    # DATASETS
    #data_name = 'skeleton_stop_walk_repeat.csv'    
    train_dataset = OrthoticDataset(train=True, transform=tr)
    test_dataset = OrthoticDataset(train=False, transform=tr)
    
    train_loader = DataLoader(dataset=train_dataset,
         batch_size=args.batch_size,
         shuffle=False,
         num_workers=32)

    # Training Parameters
    device = torch.device('cuda:{}'.format(args.gpu_ids)) if torch.cuda.is_available() else torch.device('cpu') 
    lr = args.learning_rate
    epochs = args.epochs
    
    # Models
    if args.model == 'lstm':
        net = LSTMRegressor(num_gyro, orthotic_height * orthotic_width * 2).to(device)
    elif args.model == 'linear':
        net = LinearRegressor(num_gyro*seq_len, orthotic_height * orthotic_width * 2).to(device)

    args.loss = args.loss.lower()
    if args.loss == 'rmse':
        criterion = nn.RMSELoss() 
    # mse
    else :
        criterion = nn.MSELoss()
   
    #optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    #optim = torch.optim.Adam(net.parameters(), lr=lr)
    optim = torch.optim.AdamW(net.parameters(), lr=lr)
    
    # train
    for epoch in range(1, epochs+1):
        net.train()
        train_loss = 0
        train_len = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            if args.model == 'linear':
                x= x.reshape(x.size(0), x.size(1)*x.size(2))
                pred = net(x)
            elif args.model == 'lstm':
                hc = net.init_hidden_cell(args.batch_size)
                pred, hc = net(x, hc)

            loss = criterion(pred, y)

            loss.backward()
            optim.step()
            optim.zero_grad()
    
            train_loss += loss
            train_len += len(x)
    
        print('epoch: {}'.format(epoch))
        print('tr_loss: {}'.format(train_loss.item() / train_len))
   

    print('Save Model ...')
    model_dir = 'logs'
    model_file = 'orthotics_{}_b{}_e{}_lr{}_{}.pt'.format(args.model, 
            args.batch_size, args.epochs, str(args.learning_rate).replace('.','_'), args.loss)
    model_path = os.path.join(model_dir, model_file)
    torch.save(net.state_dict(), model_path)
    print(model_path, 'Saved.')  
   
    # Test (Prediction)
    print(test_dataset.y.size())
    test_y = test_dataset.y.view(-1, orthotic_height*2, orthotic_width)
    left = test_dataset.y[:, :orthotic_height]
    right = test_dataset.y[:, orthotic_height:] 

    test_pred = orthotics(gyro_data=test_dataset.x, orthotic_left=left, orthotic_right=right, model_type=args.model,
            model_dir=model_dir, model_file=model_file, batch_size=args.batch_size, gpu_ids=args.gpu_ids)
    test_pred = torch.Tensor(test_pred)
         
