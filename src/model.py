import torch.nn as nn
import torch


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.stddev
            return x * (1 + noise)

        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, r=8):
        super().__init__()
        # takes input of shape BxCxT
        # purpose is to squeeze the temporal dimension and excite the Channels dimension
        self.excitation_net = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # x: [B x C x T]
        squeezed = torch.mean(x, dim=2)
        weights = self.excitation_net(squeezed).unsqueeze(-1)  # B x C x 1
        out = x * weights
        return out


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, r=8):
        super().__init__()
        self.score_generator = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        """
        expected x shape: B x T x C
        """
        scores = self.score_generator(x)  # Shape: (B, T, 1)
        out = torch.sum(scores * x, dim=1)
        return out


class ReSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            #
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.se_block = SEBlock(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut_layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.Dropout1d(0.3),
            )
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Identity(),
            )

        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        shortcut = self.shortcut_layer(x)
        out = self.conv_block(x)
        out = self.se_block(out)

        out += shortcut
        out = self.final_relu(out)

        return out


class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, layer_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: batch x seq x features
        """
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm_layer(x, (h0, c0))  # B x T x 2 * hidden_dim
        return out


class GRUBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, layer_dim=1):
        super().__init__()
        self.gru_layer = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: batch x seq x features
        """
        out, h = self.gru_layer(x)  # B x T x 2 * hidden_dim
        return out


# Keeping this for refernece
class FullModel(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.noise_layer = nn.Sequential(
            GaussianNoise(stddev=0.09),
        )
        self.res_connection = nn.Sequential(
            ReSEBlock(in_channels, 64, stride=2),
            # ReSEBlock(64, 64, stride=2),
            ReSEBlock(64, 128, stride=2),
        )
        self.lstm_layer = LSTMBlock(128, hidden_dim=128, layer_dim=1)  # 64 + 64 // 2
        self.gru_layer = GRUBlock(128, hidden_dim=128, layer_dim=1)

        self.temporal_attention = nn.Sequential(
            AttentionBlock(in_channels=4 * 128),
        )

        self.shortcut_branch = nn.Sequential(
            GaussianNoise(stddev=0.09),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 16),
            nn.ELU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(4 * 128 + 16, 256),  # Match author's layers
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUBlock(6).to(device)
    dummy = torch.randn(2, 6)
    print(model(dummy))
