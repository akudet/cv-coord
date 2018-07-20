import torch.nn as nn


class LSTMConv(nn.Module):

    def __init__(self, in_feature, out_features, kernel_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(in_feature, hidden_size)
        self.fc = nn.Linear(hidden_size, out_features)
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.out_features = out_features

    def forward(self, feats):
        N, C, H, W = feats.shape
        HH, WW = self.kernel_size
        DH, DW = self.stride
        H_out = 1 + (H - HH) // DH
        W_out = 1 + (W - WW) // DW

        out = feats.new_empty((N, self.out_features, H_out, W_out))
        feats = feats.transpose(1, 2).transpose(2, 3)
        for i in range(H_out):
            for j in range(W_out):
                r, c = i * DH, j * DW
                field = feats[:, r:r + HH, c:c + WW].reshape(N, -1, C)
                h, states = self.lstm(field)
                out[:, :, i, j] = self.fc(h[:, -1, :])
        return out


class Flatten(nn.Module):

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


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
