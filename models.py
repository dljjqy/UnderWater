import torch
import torch.nn as nn
from random import random

class Encoder(nn.Module):
    def __init__(self, features, emb_dim, hide_size, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Linear(features, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hide_size, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout((self.embedding(x)))
        y, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, features, emb_dim, hide_size, n_layers, dropout):
        super().__init__()
        self.act = nn.ReLU()
        self.embedding = nn.Linear(features, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hide_size, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hide_size, features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h, c):
        embedded = self.dropout((self.embedding(x)))
        output, (hidden, cell) = self.rnn(embedded, (h, c))
        prediction = self.fc_out(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, features, hidsize=256, Eembsize=128, Dembsize=128, n_layers=4, dropout=0):
        '''
        features: How many features in a single row.
        EhidSize: The hiden size of Encoder.
        DhidSize: The hiden size of Decoder.
        Eembsize: The embed size of Encoder.
        Dembsize: The embed size of Decoder.
        n_layers: The layer of LSTM.
        dropout: The ratio of dropout.
        '''
        super().__init__()
        self.encoder = Encoder(features, Eembsize, hidsize, n_layers, dropout)
        self.decoder = Decoder(features, Dembsize, hidsize, n_layers, dropout)

    def forward(self, x, y, teach_forcing_ratio = 0.5):
        '''
        x: lGet x Features
        y: lPre x Features
        '''
        lPre = y.shape[0]
        previous = x[-1:, :, :]
        predictions = torch.zeros_like(y).type_as(y)
        h, c = self.encoder(x)

        for t in range(lPre):
            pre, h, c = self.decoder(previous, h, c)
            predictions[t] = pre
            teacher_force = random() < teach_forcing_ratio
            if teacher_force:
                previous = y[t, :, :]
                previous = previous.unsqueeze(0)
            else:
                previous = pre
        return predictions
        
class Splitting(nn.Module):
    def __init__(self):
        super().__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        return (self.even(x), self.odd(x))

class Interactor(nn.Module):
    def __init__(self, in_planes, splitting=True, kernel=5, dropout=0.5, groups=1, hidden_size=1, INN = True):
        super(Interactor, self).__init__()
        self.modified = INN
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups

        self.splitting = splitting
        self.split = Splitting()

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1



if __name__ == '__main__':
    features = 3
    batch_size = 8 
    lGet = 24
    lPre = 6
    x = torch.rand(lGet, batch_size, features)
    y = torch.rand(lPre, batch_size, features)
    net = Seq2Seq(features)
    print(net(x, y).shape)
