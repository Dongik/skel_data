import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
import torch.nn.functional as F

from dataset import SkelDataset
from model import FC

#tr = torch.nn.Sequential(
#    torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
tr = None

#data_name = 'skeleton_stop_walk_repeat.csv'

train_dataset = SkelDataset(train=True, transform=tr)
test_dataset = SkelDataset(train=False, transform=tr)

train_loader = DataLoader(dataset=train_dataset,
     batch_size=128,
     shuffle=True,
     num_workers=32)
test_loader = DataLoader(dataset=test_dataset,
    batch_size=128,
     shuffle=False,
     num_workers=32)

device = torch.device('cuda:1')
lr = 0.0001
epochs = 200

net = FC().to(device)
criterion = nn.MSELoss()
#criterion = lambda pred, y: torch.mean((pred - y)**2)
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
    print('tr_loss: {}'.format(train_loss.item()))

# test
net.eval()
with torch.no_grad():
    test_loss = 0
    test_len = 0
    test_pred = torch.empty(0,102)
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        pred = net(x)
        loss = criterion(pred, y)

        test_loss += loss
        test_len += len(x)
        test_pred = torch.cat([test_pred, pred.cpu()])

print('test loss: {}'.format(test_loss))

# save csv
test_pred = torch.cat([torch.zeros(len(test_pred),84), test_pred], dim=1)
test_y = torch.cat([torch.zeros(len(test_dataset.y),84), test_dataset.y], dim=1)

df_pred = pd.DataFrame(test_pred.numpy())
df_y = pd.DataFrame(test_y.numpy())

print('Save Model ...')
torch.save(net.state_dict(), 'model.pt')

print('Save CSVs ...')
df_pred.to_csv('skel_pred.csv', sep=',', index=False)
df_y.to_csv('skel_y.csv', sep=',', index=False)

