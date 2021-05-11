import os
import pandas as pd
import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms

num_gyro = 44
num_skel = 51
orthotic_width = 10
orthotic_height = 30
orthotic_train_num = 5000 # must be even

# x: foot pressure, y: skel
class SkelDataset(Dataset):
    def __init__(self, train=True, data_x = None, data_y = None, csv_dir='skeleton_data', csv_file=None, train_ratio=0.8, transform=None):
        # data array exist
        if data_x is not None:
            self.x = torch.Tensor(data_x)
            if data_y is not None:
                self.y = torch.Tensor(data_y)
            else:
                self.y = torch.zeros(len(data_x), num_skel)
        
        # Read csv file
        else:
            # Read whole directory
            if csv_file is None:
                files = glob.glob(os.path.join(csv_dir, '*.csv'))
            # Read specific csv file
            else:
                files = [os.path.join(csv_dir, csv_file)]
            
            x, y = [], [] 
            for file_name in files:
                # read file
                print('read file {}'.format(file_name))
                df = pd.read_csv(file_name, index_col=0)

                # split dataset (each file)
                pivot = int(len(df) * train_ratio)
                if train:
                    x_, y_ = df.iloc[:pivot, :44], df.iloc[:pivot, 44:]
                else:
                    x_, y_ = df.iloc[pivot:, :44], df.iloc[pivot:, 44:]
                x.append(torch.Tensor(x_.values))
                y.append(torch.Tensor(y_.values))

            # concat to one tensor 
            self.x, self.y = torch.cat(x, dim=0), torch.cat(y, dim=0) 

        print("X:", self.x.size())
        print("Y:", self.y.size())

        self.transform = transform
        
    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx], self.y[idx]

# x: foot pressure, y: skel
class SkelSeqDataset(Dataset):
    def __init__(self, train=True, data_x = None, data_y = None, seq_len=200, csv_dir='skeleton_data', csv_file=None, train_ratio=0.8, transform=None):
        # data array exist
        if data_x is not None:
            self.x = torch.Tensor(data_x)  
            if data_y is not None:
                self.y = torch.Tensor(data_y) 
            else:
                self.y = torch.zeros(len(data_x) // seq_len, seq_len, num_skel)
        
        # Read csv file
        else:
            # Read whole directory
            if csv_file is None:
                files = glob.glob(os.path.join(csv_dir, '*.csv'))
            # Read specific csv file
            else:
                files = [os.path.join(csv_dir, csv_file)]
            
            x, y = [], [] 
            for file_name in files:
                # read file
                print('read file {}'.format(file_name))
                df = pd.read_csv(file_name, index_col=0)
                
                # split dataset (each file)
                pivot = int(len(df) * train_ratio)
                if train:
                    df_ = df.iloc[:pivot-(pivot%seq_len)]
                else:
                    df_ = df.iloc[pivot:]
                    df_ = df_.iloc[:len(df_)-(len(df_)%seq_len)] 
                 
                x.append(torch.Tensor(df_.iloc[:, :44].values))
                y.append(torch.Tensor(df_.iloc[:, 44:].values))
            # concat to one tensor 
            self.x, self.y = torch.cat(x, dim=0), torch.cat(y, dim=0) 
        
        # reshape by seq_len
        if len(self.x.size()) < 3:
            pivot = self.x.size(0) % seq_len
            self.x = self.x[:self.x.size(0)-pivot]
            self.x = self.x.reshape(self.x.size(0) // seq_len, seq_len, self.x.size(1))
        if len(self.y.size()) < 3:
            pivot = self.y.size(0) % seq_len
            self.y = self.y[:self.y.size(0)-pivot]
            self.y = self.y.reshape(self.y.size(0) // seq_len, seq_len, self.y.size(1))
        print("X:", self.x.size())
        print("Y:", self.y.size())

        self.seq_len = seq_len
        self.transform = transform
        
    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx], self.y[idx]

# x: foot pressure, y: orthotics(left&right)
class OrthoticDataset(Dataset):
    def __init__(self, train=True, data_x = None, data_y = None, data_dir='orthotics_data', train_ratio=0.8, transform=None):
        # data array exist
        if data_x is not None:
            self.x = torch.Tensor(data_x) 
            if data_y is not None:
                self.y = torch.Tensor(data_y)
            else:
                self.y = torch.zeros(len(data_x), orthotic_height*orthotic_width*2)
        
        # Read csv file
        else:
            self.x, self.y = read_orthotics_data(data_dir) 
        
            # Split dataset if Train
            pivot = int(len(self.y) * train_ratio)
            if train:
                self.x = self.x[:pivot]
                self.y = self.y[:pivot]
            else:
                self.x = self.x[pivot:]
                self.y = self.y[pivot:]

        print("X:", self.x.size())
        print("Y:", self.y.size())

        self.transform = transform
        # Normalize RGB Value(0~255)
        self.y = self.y / 256

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx], self.y[idx] 

# make Tensor and resize to fixed size(height*width)
def totensor_orthotics(df):
    transform = transforms.Resize((orthotic_height, orthotic_width))
    t = torch.Tensor(df.values)
    t = torch.unsqueeze(t, dim=0)
    t = transform(t)
    #t = torch.squeeze(t, dim=0)
    return torch.flatten(t)

def read_orthotics_data(data_dir):
    x, y = [], []
    for path, dirs, files in os.walk(data_dir):
        y_ = torch.zeros(orthotic_height * 2 * orthotic_width)
        for filename in files:
            # read gyro data(x)
            if filename == 'gyro_pressure.csv' != -1:
                gyro_ = pd.read_csv(os.path.join(path, filename), index_col=0)
                # use middle 5000 data
                gyro_num = len(gyro_)
                if gyro_num > orthotic_train_num:
                    gyro_ = gyro_[(gyro_num//2)-(orthotic_train_num//2):(gyro_num//2)+(orthotic_train_num//2)]
                # repeat when data length < 5000
                else:
                    gyro_ = pd.concat([gyro_]*(orthotic_train_num//gyro_num + 1))[:orthotic_train_num]
                x.append(torch.Tensor(gyro_.values))
            # read left orthotic
            elif filename == 'left_orthotics.csv':
                left_ = pd.read_csv(os.path.join(path, filename), index_col=0)
                y_[:orthotic_height * orthotic_width] = totensor_orthotics(left_)
            elif filename == 'right_orthotics.csv':
                right_ = pd.read_csv(os.path.join(path, filename), index_col=0)
                y_[orthotic_height * orthotic_width:] = totensor_orthotics(right_)

        # append when orthotics readed
        if torch.sum(y_!=0).item() > 0:
            y.append(y_)
    return torch.stack(x, dim=0), torch.stack(y, dim=0)

if __name__=='__main__':
    print('Loading SkelDataset...')
    skel_train = SkelDataset(train=True)
    skel_test = SkelDataset(train=False)
    print('Loading SkelSeqDataset...')
    skel_train = SkelSeqDataset(train=True)
    skel_test = SkelSeqDataset(train=False)
    print('Loading OrthoticDataset...')
    orthotic_train = OrthoticDataset(train=True)
    orthotic_test = OrthoticDataset(train=False)
