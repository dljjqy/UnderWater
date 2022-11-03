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
    def __init__(self, features, hidsize=512, Eembsize=256, Dembsize=256, n_layers=4, dropout=0):
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
        return x[:, :, ::2]

    def odd(self, x):
        return x[:, :, 1::2]

    def forward(self, x):
        return (self.even(x), self.odd(x))

class OneDim_DoubleConv(nn.Module):
    def __init__(self, in_c, hid_c, out_c, kernel_size=5, stride=1, dilation=1, dropout=0.3):
        super().__init__()
        parts = [
            nn.Conv1d(in_c, hid_c, kernel_size, stride, 
                padding = 'same', padding_mode = 'replicate', dilation = dilation),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hid_c, out_c, 3, stride, 
                padding = 'same', padding_mode = 'replicate', dilation = dilation),
            nn.Tanh()]
        self.net = nn.Sequential(*parts)

    def forward(self, x):
        return self.net(x)


class Interactor(nn.Module):
    def __init__(self, features, hidden_size_rate=2):
        super(Interactor, self).__init__()
        self.splitting = Splitting()
        self.phi = OneDim_DoubleConv(features, hidden_size_rate * features, features)
        self.psi = OneDim_DoubleConv(features, hidden_size_rate * features, features)
        self.rho = OneDim_DoubleConv(features, hidden_size_rate * features, features)
        self.yita= OneDim_DoubleConv(features, hidden_size_rate * features, features)

    def forward(self, x):
        even, odd = self.splitting(x)
        x = torch.cat((even, odd), 2).mul(
                torch.cat((torch.exp(self.psi(odd)), 
                           torch.exp(self.phi(even))), 2))
        
        evens, odds = self.splitting(x)
        x = torch.cat((odds, evens), 2) + \
                torch.cat((self.rho(evens), -self.yita(odds)), 2)
        return self.splitting(x)

class SciNet_Tree(nn.Module):
    def __init__(self, features, levels, hidden_size_rate):
        super().__init__()
        self.level = levels
        self.block = Interactor(features, hidden_size_rate)
        if self.level != 0:
            self.even  = SciNet_Tree(features, self.level-1, hidden_size_rate)
            self.odd = SciNet_Tree(features, self.level-1, hidden_size_rate)

    def _zip(self, x_even, x_odd):
        N, F, Leven = x_even.shape
        Lodd = x_odd.shape[-1]

        y = torch.zeros(N, F, Leven + Lodd).type_as(x_even)
        y[..., ::2] += x_even
        y[..., 1::2] += x_odd
        return y

    def forward(self, x):
        x_even, x_odd = self.block(x)
        if self.level == 0:
            return self._zip(x_even, x_odd)
        else:
            return self._zip(self.even(x_even), self.odd(x_odd))

class SCINet(nn.Module):
    def __init__(self, features, lPre, lGet, Tree_levels=2, hidden_size_rate=1):
        super().__init__()
        self.encoder = SciNet_Tree(features, Tree_levels, hidden_size_rate)
        self.decoder = nn.Conv1d(lGet, lPre, 1, 1, bias=False)

    def forward(self, x):
        y = self.encoder(x)
        y += x
        y = torch.transpose(y, 1, 2)
        y = self.decoder(y)
        y = torch.transpose(y, 1, 2)
        return y


if __name__ == '__main__':
    features = 9
    batch_size = 8 
    lPre = 42
    lGet = 2 * lPre

    x = torch.rand(batch_size, features, lGet)
    net = SCINet(features, lPre, lGet)
    y = net(x)
    print(x.shape)
    print(y.shape)
    


