import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hide_dim, n_layers, dropout):
        '''
        input_dim: The dimension of input vector.
        emb_dim: Embedding dimension.
        hide_dim: Dimension of the hidden and cell states.
        n_layers: The number of layers in RNN
        '''
        super().__init__()
        self.embedding = nn.Linear(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hide_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        y, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

