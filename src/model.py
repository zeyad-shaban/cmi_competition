import torch.nn as nn
import torch

class SimpleModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(6, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, n_classes),
        )

    def forward(self, x: torch.Tensor):
        y_pred = self.fc(x)

        return y_pred


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleModel(6).to(device)
    dummy = torch.randn(2, 6)
    print(model(dummy))