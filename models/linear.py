from torch import nn
from torch.nn import functional as F

# Many to Many(Sequential)
class LinearRegressor(nn.Module):
    def __init__(self, input_dim=44, output_dim=51):
        super(LinearRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128) 
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 384)

        self.regressor = nn.Linear(384, output_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(384)
		 
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.regressor(x)
        return x

    def init_weights(self):
        # Weight initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight) 
        nn.init.xavier_uniform_(self.regressor.weight)

if __name__=="__main__":
    import torch
    batch_size, input_dim, output_dim = 8, 44, 51

    net = LinearRegressor(input_dim, output_dim)

    x = torch.ones(batch_size, input_dim)
    print('Input Dim: {}'.format(x.size()))

    output = net(x)
    print('Output Dim: {}'.format(output.size()))

