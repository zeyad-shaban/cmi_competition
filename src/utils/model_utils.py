import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import torch
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np

if __name__ == "__main__":
    from math_utils import resize_spectrograms_torch
else:
    from src.utils.math_utils import resize_spectrograms_torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaulate_model(y_pred, y_true, target_gestures_encoded, encoder: LabelEncoder):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(target_gestures_encoded, torch.Tensor):
        target_gestures_encoded = target_gestures_encoded.cpu().numpy()

    conf_matrix_result = confusion_matrix(y_true, y_pred)
    clsf_report_result = pd.DataFrame(classification_report(y_true, y_pred, target_names=encoder.classes_, output_dict=True, zero_division=0)).T

    y_true_binary = np.isin(y_true, target_gestures_encoded)
    y_pred_binary = np.isin(y_pred, target_gestures_encoded)
    f1_binary = f1_score(y_true_binary, y_pred_binary, average="binary", zero_division=0)

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    competition_evaluation = 0.5 * f1_binary + 0.5 * f1_macro

    return {
        "confusion_matrix": conf_matrix_result,
        "classification_report": clsf_report_result,
        "f1_binary": f1_binary,
        "f1_macro": f1_macro,
        "competition_evaluation": competition_evaluation,
    }


def predict_in_chunks(model, X, device, batch_size=10):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = resize_spectrograms_torch(X[i : i + batch_size]).to(device)
            preds = torch.argmax(model(batch), dim=1).cpu()
            preds_list.append(preds)
    return torch.cat(preds_list)


def extract_features_in_chunks(model, X, device, batch_size=10):
    model = model.to(device)
    model.eval()
    features_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = resize_spectrograms_torch(X[i : i + batch_size]).to(device)
            feats = model(batch).view(batch.shape[0], -1).cpu()
            features_list.append(feats)
    return torch.cat(features_list)


def train_model(model: nn.Module, dataloader: DataLoader, n_epochs: int, should_log=True, spectogram_features=False, specto_size=(224, 224)):
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(n_epochs):
        loss_avg = 0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            if spectogram_features:
                x = resize_spectrograms_torch(x, specto_size)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss_avg += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        loss_avg = loss_avg / len(dataloader)
        if (epoch) % 20 == 0 and should_log:
            print(f"{epoch} - loss_avg: {loss_avg}")


if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader

    class SimpleModel(nn.Module):
        def __init__(self, n_classes):
            super().__init__()

            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(17 * 224 * 224, n_classes),
            )

        def forward(self, x: torch.Tensor):
            y_pred = self.fc(x)

            return y_pred

    n_classes = 6
    dummy_features = torch.randn((64, 17, 9, 9), dtype=torch.float32)
    dummy_target = torch.randint(0, n_classes, (64,), dtype=torch.long)
    dummy_encoder = LabelEncoder()

    # encoding
    dummy_encoder.fit(dummy_target.numpy())
    target_gestures_encoded = torch.arange(0, n_classes)
    print(target_gestures_encoded)

    # training
    model = SimpleModel(n_classes)
    dataloader = DataLoader(TensorDataset(dummy_features, dummy_target), batch_size=16, shuffle=True)
    train_model(model, dataloader, n_epochs=2, spectogram_features=True)

    # evaluating
    y_pred = model(resize_spectrograms_torch(resize_spectrograms_torch(dummy_features))).argmax(dim=1)
    print(evaulate_model(y_pred, dummy_target, target_gestures_encoded, dummy_encoder))
