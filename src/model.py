import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.model = nn.Sequential(
            nn.Linear(4, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2),
        )

    def forward(self, x):
        return self.model(x)


class LSTMModel(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(2, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 2)
        h, _ = self.lstm(x)
        y = self.fc(h[:, -1])
        return y
