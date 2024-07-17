import torch
import torch.nn as nn
import torch.optim as optim


class HandGestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(HandGestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x[:, -1, :]))
        x = self.fc2(x)
        return x
