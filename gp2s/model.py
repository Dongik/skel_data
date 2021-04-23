from torch import nn
from torch.nn import functional as F

class FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(44, 128) 
        self.fc2 = nn.Linear(128, 384)
        self.fc3 = nn.Linear(384, 102)
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(384)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight) 

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.fc3(x)
        return x
