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


def evaluate_model(y_pred, y_true, target_gestures_encoded, encoder: LabelEncoder):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(target_gestures_encoded, torch.Tensor):
        target_gestures_encoded = target_gestures_encoded.cpu().numpy()

    # Calculate binary metrics first (before modification)
    y_true_binary = np.isin(y_true, target_gestures_encoded)
    y_pred_binary = np.isin(y_pred, target_gestures_encoded)
    f1_binary = f1_score(y_true_binary, y_pred_binary, average="binary", zero_division=0)

    # Create copies for modification (to avoid changing original arrays)
    y_true_modified = y_true.copy()
    y_pred_modified = y_pred.copy()

    # Combine non-target gestures into a single class
    new_class_id = np.max(target_gestures_encoded) + 1
    y_pred_modified[~np.isin(y_pred_modified, target_gestures_encoded)] = new_class_id
    y_true_modified[~np.isin(y_true_modified, target_gestures_encoded)] = new_class_id

    # Create target names for the modified classes
    target_names_modified = []

    # Add names for target gestures (in the order they appear)
    unique_classes = np.unique(np.concatenate([y_true_modified, y_pred_modified]))
    for class_id in unique_classes:
        if class_id in target_gestures_encoded:
            # Use original name from encoder
            target_names_modified.append(encoder.classes_[class_id])
        else:
            # This is our combined non-target class
            target_names_modified.append("Non-target")

    conf_matrix_result = confusion_matrix(y_true_modified, y_pred_modified)
    clsf_report_result = pd.DataFrame(classification_report(y_true_modified, y_pred_modified, target_names=target_names_modified, output_dict=True, zero_division=0)).T

    f1_macro = f1_score(y_true_modified, y_pred_modified, average="macro", zero_division=0)
    competition_evaluation = 0.5 * f1_binary + 0.5 * f1_macro

    return {
        "confusion_matrix": conf_matrix_result,
        "classification_report": clsf_report_result,
        "f1_binary": f1_binary,
        "f1_macro": f1_macro,
        "competition_evaluation": competition_evaluation,
    }


def predict_in_chunks(model, X, device, batch_size):
    preds_list = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = resize_spectrograms_torch(X[i : i + batch_size])
            batch = batch.to(device, non_blocking=True)
            preds = torch.argmax(model(batch), dim=1).cpu()
            preds_list.append(preds)
            del batch  # free right after use
            torch.cuda.empty_cache()
    return torch.cat(preds_list)


def extract_features_in_chunks(model, X, device, batch_size=1024):
    model.eval()
    features_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = resize_spectrograms_torch(X[i : i + batch_size]).to(device)
            feats = model(batch).view(batch.size(0), -1).cpu()
            features_list.append(feats)
            del batch, feats
            torch.cuda.empty_cache()
    return torch.cat(features_list, dim=0).numpy()


def train_model(model: nn.Module, dataloader: DataLoader, n_epochs: int, should_log=True, mixup_alpha=0.4, lr=5e-3, weight_decay=3e-3, n_classes=9):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Add learning rate scheduler - reduces LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20)

    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(n_epochs):
        loss_avg = 0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            lam = np.random.beta(mixup_alpha, mixup_alpha)
            # ---- MIXUP IMPLEMENTATION ----
            batch_size = x.shape[0]
            shuffled_indices = torch.randperm(batch_size)

            x_batch_b = x[shuffled_indices]
            y_batch_b = y[shuffled_indices]

            mixed_x = lam * x + (1 - lam) * x_batch_b
            y_one_hot = nn.functional.one_hot(y, num_classes=n_classes).float()
            y_b_one_hot = nn.functional.one_hot(y_batch_b, num_classes=n_classes).float()
            mixed_y = lam * y_one_hot + (1 - lam) * y_b_one_hot

            # ---- MIXUP ENDS ----

            y_pred = model(mixed_x)
            loss = criterion(y_pred, mixed_y)
            loss_avg += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        loss_avg = loss_avg / len(dataloader)

        # Step the scheduler with the current loss
        scheduler.step(loss_avg)

        if (epoch) % 20 == 0 and should_log:
            current_lr = opt.param_groups[0]["lr"]
            print(f"{epoch} - loss_avg: {loss_avg:.4f}, lr: {current_lr:.6f}")


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
    print(evaluate_model(y_pred, dummy_target, target_gestures_encoded, dummy_encoder))
