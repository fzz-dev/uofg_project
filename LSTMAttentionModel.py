import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.context_vector = nn.Linear(hidden_size * 2, 1, bias=False)

    def forward(self, lstm_output):
        attn_weights = torch.tanh(self.attention(lstm_output))
        attn_weights = self.context_vector(attn_weights).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights.unsqueeze(-1) * lstm_output, dim=1)
        return context, attn_weights

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, dropout_rate=0.3):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.attention = Attention(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        context = self.dropout(context)
        context = self.batch_norm(context)
        out = self.fc(context)
        return out



