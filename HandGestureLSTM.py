import torch
import torch.nn as nn
import torch.optim as optim


# 定义手势识别模型
class HandGestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HandGestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])  # 使用最后一个时间步的输出
        return x
