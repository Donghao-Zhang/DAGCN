import torch.nn as nn


class LSTM_decorator(nn.Module):
    def __init__(self, in_dim, hidden_dim, layers, dropout=None):
        super(LSTM_decorator, self).__init__()
        self.layers = layers
        lstm = [nn.LSTM(in_dim, hidden_dim, 1, batch_first=True, bidirectional=True)]
        for i in range(layers-1):
            lstm += [nn.LSTM(hidden_dim*2, hidden_dim, 1, batch_first=True, bidirectional=True)]
        self.lstm = nn.ModuleList(lstm)
        if dropout is not None and layers > 1:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, masks, pad_idx=0):
        seq_lens = list(masks.data.eq(pad_idx).long().sum(1).squeeze())
        for i in range(self.layers):
            if i != 0 and self.dropout is not None:
                x = self.dropout(x)
            x = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True)
            x, _ = self.lstm[i](x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x
