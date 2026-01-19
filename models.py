from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn

torch.manual_seed(0)


class MLPPred(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0.05):
        super(MLPPred, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1

        self.mlp1 = nn.Linear(input_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size * 2)
        self.mlp3 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.mlp4 = nn.Linear(hidden_size * 2, output_size)
        # Activation layers (LeakyReLU defined but ReLU is used in forward).
        self.relu = nn.LeakyReLU()
        self.relu1 = nn.ReLU()

    def forward(self, input):
        # Expected input: [batch, ..., input_size] (the last dim matches input_size).
        mlp1_out = self.mlp1(input)
        mlp2_out = self.mlp2(self.relu1(mlp1_out))
        mlp3_out = self.mlp3(self.relu1(mlp2_out))
        mlp3_out = self.relu1(mlp3_out)
        output = self.mlp4(mlp3_out)
        return output


class LSTMPred(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0):
        super(LSTMPred, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = 1

        self.hidden_size = hidden_size
        self.num_layers = 1
        self.lstm1 = nn.LSTM(1, hidden_size, num_layers=self.num_layers, bidirectional=False,
                           batch_first=True,
                           dropout=dropout)
        self.bn1 = nn.BatchNorm1d(input_size)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=self.num_layers, bidirectional=False,
                           batch_first=True, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(input_size)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers=self.num_layers, bidirectional=False,
                           batch_first=True, dropout=dropout)
        self.bn3 = nn.BatchNorm1d(input_size)
        # Nonlinearity and projection heads.
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, 1)
        self.fc2 = nn.Linear(input_size, output_size)

    def forward(self, input):
        # Input: [batch, seq_len] -> expand to [batch, seq_len, 1].
        input = input.unsqueeze(2)

        lstm1_out, _ = self.lstm1(input)
        lstm1_out = self.bn1(lstm1_out)

        lstm2_out, _ = self.lstm2(self.relu(lstm1_out))
        lstm2_out = self.bn1(lstm2_out)
        lstm3_out, _ = self.lstm3(self.relu(lstm2_out))
        lstm3_out = self.bn1(lstm3_out)

        fc_out = self.fc(self.relu(lstm3_out))   # [batch, seq_len, 1]
        fc_out = fc_out.squeeze(2)               # [batch, seq_len]
        fc_out = self.fc2(self.relu(fc_out))     # [batch, output_size]
        return fc_out


class GRUPred(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0):
        super(GRUPred, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = 1
        self.gru1 = nn.GRU(1, hidden_size, num_layers=self.num_layers, bidirectional=False,
                           batch_first=True,
                           dropout=dropout)
        self.bn1 = nn.BatchNorm1d(input_size)
        self.gru2 = nn.GRU(hidden_size, hidden_size, num_layers=self.num_layers, bidirectional=False,
                           batch_first=True, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(input_size)
        self.gru3 = nn.GRU(hidden_size, hidden_size, num_layers=self.num_layers, bidirectional=False,
                           batch_first=True, dropout=dropout)
        self.bn3 = nn.BatchNorm1d(input_size)
        # Nonlinearity and projection heads.
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, 1)
        self.fc2 = nn.Linear(input_size, output_size)

    def forward(self, input):
        # Input: [batch, seq_len] -> expand to [batch, seq_len, 1].
        input = input.unsqueeze(2)

        gru1_out, _ = self.gru1(input)
        gru1_out = self.bn1(gru1_out)

        gru2_out, _ = self.gru2(self.relu(gru1_out))
        gru2_out = self.bn2(gru2_out)

        gru3_out, _ = self.gru3(self.relu(gru2_out))
        gru3_out = self.bn3(gru3_out)

        fc_out = self.fc(self.relu(gru3_out))    # [batch, seq_len, 1]
        fc_out = fc_out.squeeze(2)               # [batch, seq_len]
        fc_out = self.fc2(self.relu(fc_out))     # [batch, output_size]
        return fc_out


class RNNPred(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0):
        super(RNNPred, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = 1

        self.in2rnn = nn.Linear(input_size, hidden_size)
        self.rnn1 = nn.RNN(hidden_size, hidden_size, num_layers=self.num_layers, bidirectional=False,
                           dropout=dropout)
        self.rnn2 = nn.RNN(hidden_size, hidden_size * 2, num_layers=self.num_layers, bidirectional=False,
                           batch_first=True, dropout=dropout)
        self.rnn3 = nn.RNN(hidden_size * 2, hidden_size * 2, num_layers=self.num_layers, bidirectional=False,
                           batch_first=True, dropout=dropout)
        # Nonlinearity and output projection.
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # Input: [batch, seq_len, input_size].
        rnn1_out, _ = self.rnn1(self.in2rnn(input))
        rnn2_out, _ = self.rnn2(self.relu(rnn1_out))
        rnn3_out, _ = self.rnn3(self.relu(rnn2_out))
        fc_out = self.fc(self.relu(rnn3_out))
        return fc_out


# =========================
# V2 one-step regressor ----
# =========================
class GRUOneStepRegressor(nn.Module):
    """
    GRU-based one-step regressor.

    Input:
        x in shape [B, T, 1] (or [B, T], which is expanded to [B, T, 1])
    Output:
        y in shape [B, 1]

    Note:
        `out_activation` is optional and should match the target scaling:
        - "linear" for z-scored or unbounded targets
        - "sigmoid" for [0, 1] targets
        - "tanh" for [-1, 1] targets
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, out_activation: str = "linear"):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

        out_activation = (out_activation or "linear").lower()
        if out_activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif out_activation == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, T] -> [B, T, 1]
        out, _ = self.gru(x)      # [B, T, H]
        last = out[:, -1, :]      # [B, H]
        y = self.fc(last)         # [B, 1]
        return self.act(y)
