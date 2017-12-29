import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, D_in, D_out):
        super(Net, self).__init__()
        self.dp1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(D_in,2048)
        self.bn1 = nn.BatchNorm1d(2048)

        self.dp2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(2048,2048)
        self.bn2 = nn.BatchNorm1d(2048)

        self.dp3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(2048,2048)
        self.bn3 = nn.BatchNorm1d(2048)

        self.dp4 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(2048,2048)
        self.bn4 = nn.BatchNorm1d(2048)

        self.dp5 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(2048,D_out)
        self.bn5 = nn.BatchNorm1d(D_out)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(self.dp1(x))))
        x = F.relu(self.bn2(self.fc2(self.dp2(x))))
        x = F.relu(self.bn3(self.fc3(self.dp3(x))))
        x = F.relu(self.bn4(self.fc4(self.dp4(x))))
        return self.bn5(self.fc5(self.dp5(x)))

class Lstm(nn.Module):
    def __init__(self, D_in, hidden_dim, D_out, dropout):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(D_in, hidden_dim, dropout=dropout)
        self.hidden_dim = hidden_dim

        self.hidden2state = nn.Linear(hidden_dim, D_out)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(1, 1, self.hidden_dim)).type(torch.FloatTensor),
                Variable(torch.zeros(1, 1, self.hidden_dim)).type(torch.FloatTensor))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        return self.hidden2state(lstm_out)