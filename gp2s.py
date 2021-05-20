import pandas as pd
import numpy as np
import os
import random
import time
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
import torch.nn.functional as F

from dataset import SkelDataset, SkelSeqDataset
from models.linear import LinearRegressor
from models.lstm import LSTMRegressor

from inference import gp2s, Inference

# Remove randomness
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
#torch.backends.cudnn.deterministic = True # Calc speed decreasing issue
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

num_gyro = 44
num_skel = 51 

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float, metavar='NAME', help='Learning Rate')
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='NAME', help='Number of Epochs')
    parser.add_argument('-b', '--batch_size', default=512, type=int, metavar='NAME', help='Batch Size')
    parser.add_argument('-l', '--loss', default='mse', type=str, metavar='NAME', help='Type of Loss Function')
    parser.add_argument('-m', '--model', default='lstm', type=str, metavar='NAME', help='Type of Model')
    parser.add_argument('-seq', '--seq_len', default=64, type=int, metavar='NAME', help='Sequence Length of LSTM and Dataset')
    parser.add_argument('-str', '--stride', default=1, type=int, metavar='NAME', help='Stride(jump) of sliding window Dataset')
    parser.add_argument('-n', '--n_layers', default=2, type=int, metavar='NAME', help='A number of Layers of LSTM')
    parser.add_argument('-gpu', '--gpu_ids', default=0, type=int, metavar='NAME', help='GPU Numbers')
    args = parser.parse_args()

    #tr = torch.nn.Sequential(
    #    torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
    tr = None

    # Record Time
    start_time = time.time()

    # DATASETS
    #data_name = 'skeleton_stop_walk_repeat.csv'
    if args.model == 'linear':
        train_dataset = SkelDataset(train=True, transform=tr)
        test_dataset = SkelDataset(train=False, transform=tr) 

    elif args.model == 'lstm':
        train_dataset = SkelSeqDataset(train=True, seq_len=args.seq_len, stride=args.stride, transform=tr)
        test_dataset = SkelSeqDataset(train=False, seq_len=args.seq_len, transform=tr)
        
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=torch.get_num_threads())

    # Training Parameters
    device = torch.device('cuda:{}'.format(args.gpu_ids)) if torch.cuda.is_available() else torch.device('cpu') 
    lr = args.learning_rate
    epochs = args.epochs
    
    # Models
    if args.model == 'linear':
        net = LinearRegressor(num_gyro, num_skel).to(device)
    elif args.model == 'lstm':
        net = LSTMRegressor(num_gyro, num_skel, num_layers=args.n_layers).to(device)

    # Test Inference
    test_infer = Inference(infer_type='gp2s', model=net, model_type=args.model, batch_size=args.batch_size,
                        n_layers=args.n_layers, seq_len=args.seq_len, loss=args.loss, gpu_ids=args.gpu_ids)

    args.loss = args.loss.lower()
    if args.loss == 'rmse':
        class RMSELoss(nn.Module):
            def __init__(self):
                super(RMSELoss,self).__init__()
                self.mse = nn.MSELoss()
                self.eps = 1e-7
            def forward(self,y,y_hat):
                return torch.sqrt(self.mse(y,y_hat) + self.eps)
        criterion = RMSELoss() 
    # mse
    else :
        criterion = nn.MSELoss()

    #optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    
    # Record Time
    initialize_time = time.time()

    # train
    for epoch in range(1, epochs+1):
        net.train()
        train_loss = 0
        train_len = 0
        for x, y in train_loader:
            batch_size = len(y)
            x, y = x.to(device), y.to(device)
             
            if args.model == 'linear':
                pred = net(x)
            elif args.model == 'lstm':
                hc = net.init_hidden_cell(batch_size)
                pred, hc = net(x, hc)
            
            loss = criterion(pred, y)
                
            loss.backward()
            optim.step()
            optim.zero_grad()
    
            train_loss += loss
            train_len += len(x)
    
        print('epoch: {}'.format(epoch))
        print('tr_loss: {}'.format(train_loss.item() / train_len))
        
        # test(val)
        if epoch % 10 == 0:
            test_pred = test_infer.infer(gyro_data=test_dataset.x, skel_data=test_dataset.y)
            #test_pred = gp2s(gyro_data=test_dataset.x, skel_data=test_dataset.y, seq_len=args.seq_len, 
            #        model_type=args.model, model=net, batch_size=args.batch_size, n_layers=args.n_layers, gpu_ids=args.gpu_ids)
    
    # Records Time
    end_time = time.time()
    train_elapsed = end_time - initialize_time
    initial_elapsed = initialize_time - start_time
    print('Initializing Time: %dm %.2fs' % (initial_elapsed // 60, initial_elapsed % 60))
    print('Training Time: %dm %.2fs' % (train_elapsed // 60, train_elapsed % 60))

    # Save Model
    print('Save Model ...')
    model_dir = 'logs'
    model_file = 'gp2s_{}_e{}_n{}_b{}_lr{}_seq{}_str{}_{}.pt'.format(args.model, args.epochs, args.n_layers,
            args.batch_size, str(args.learning_rate).replace('.','_'), args.seq_len, args.stride, args.loss)
    model_path = os.path.join(model_dir, model_file)
    torch.save(net.state_dict(), model_path)
    print(model_path, 'Saved.')
    

    # Test (Prediction)
    test_pred = test_infer.infer(gyro_data=test_dataset.x, skel_data=test_dataset.y)
    #test_pred = gp2s(gyro_data=test_dataset.x, skel_data=test_dataset.y, seq_len=args.seq_len, model_type=args.model,
    #        model_dir=model_dir, model_file=model_file, batch_size=args.batch_size, n_layers=args.n_layers, gpu_ids=args.gpu_ids)
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
