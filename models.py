import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class LSTMModel(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, output_size=1, num_layers=1, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        return self.fc(last_out)


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, output_size=1, dropout=0.3):
        super(CNNLSTMModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=input_size * 16, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, features). Conv1d expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # -> (batch, features, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.size(0), 1, -1)
        lstm_out, _ = self.lstm(x)
        last_out = self.dropout(lstm_out[:, -1, :])
        return self.fc(last_out)



class LSTMRegressorSklearn(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_size=64, lr=0.003, dropout=0.3, epochs=30):
        self.hidden_size = hidden_size
        self.lr = lr
        self.dropout = dropout
        self.epochs = epochs
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y):
        self.model = LSTMModel(hidden_size=self.hidden_size, dropout=self.dropout).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        X_tensor = torch.tensor(X[:, np.newaxis, :], dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=8, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.model(x_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X[:, np.newaxis, :], dtype=torch.float32).to(self.device)
            pred = self.model(X_tensor).cpu().numpy().squeeze()
        return pred