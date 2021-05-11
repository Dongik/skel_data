from torch import nn
from torch.nn import functional as F

orthotic_width = 10
orthotic_height = 30

# Many to One
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=44, output_dim=orthotic_width*orthotic_height, hidden_dim=256, num_layers=6, drop_prob=0.5, mult_pred=False):
        super(LSTMRegressor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.mult_pred = mult_pred

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=drop_prob, batch_first=True)
        self.regressor = nn.Linear(hidden_dim, output_dim)

        self.init_weights()

    def forward(self, x, hc=None):
        batch_size = x.size(0)

        x, hc = self.lstm(x, hc)
        
        # get whole output
        if self.mult_pred:
            x = x.reshape(x.size(0)*x.size(1), self.hidden_dim)
        # get last output
        else:
            x = x[:,-1]

        x = self.regressor(x)
        x = x.reshape(batch_size, x.size(0)//batch_size, x.size(1))
        return x, hc

    def init_weights(self):
        # Weight initialization
        nn.init.xavier_uniform_(self.regressor.weight)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def init_hidden_cell(self, batch_size=1):
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())

if __name__=="__main__":
    import torch
    batch_size, seq_len, input_dim, output_dim = 8, 1, 44, 600

    net = LSTMRegressor(input_dim, output_dim)

    x = torch.ones(batch_size, seq_len, input_dim)
    hc = net.init_hidden_cell(batch_size)

    print('Input Dim: {}'.format(x.size()))
    print('H Dim: ({}, {})'.format(hc[0].size(), hc[1].size()))

    output, hc = net(x, hc)
    print('Output Dim: {}'.format(output.size()))
    print('H Dim: ({}, {})'.format(hc[0].size(), hc[1].size()))
