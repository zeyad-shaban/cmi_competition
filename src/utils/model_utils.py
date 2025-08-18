import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import torch
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_tensor(X: torch.Tensor | np.ndarray) -> torch.Tensor:
    # Normalize
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    mean = X.mean(dim=(0, 2)).view(1, -1, 1)  # shape (1xFx1)
    std = X.std(dim=(0, 2), unbiased=False).view(1, -1, 1) + 1e-8  # shape (1xFx1)
    X = (X - mean) / (std + 1e-14)

    return X


def evaulate_model(y_pred, y_true, target_gestures_encoded, encoder: LabelEncoder):
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

def evaluate_fold(model, X, y, target_gestures_encoded, encoder, feature_mask=None):
    """Helper to evaluate model on data with optional feature masking"""
    if feature_mask is not None:
        X_masked = X.clone()
        X_masked[:, feature_mask:, :] = -1
        y_pred = torch.argmax(model(X_masked.to(device)), dim=1)
    else:
        y_pred = torch.argmax(model(X.to(device)), dim=1)
    
    return evaulate_model(y_pred.to(device), y.to(device), target_gestures_encoded, encoder)

def store_results(results_dict, score_dashboard):
    """Helper to store evaluation results"""
    for key in ['f1_macro', 'f1_binary', 'competition_evaluation', 'confusion_matrix']:
        if key not in results_dict:
            results_dict[key] = []
        results_dict[key].append(score_dashboard[key])


def train_model(model: nn.Module, dataloader: DataLoader, n_epochs: int, should_log=True, lr=5e-3, weight_decay=3e-3):
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

            y_pred = model(x)
            loss = criterion(y_pred, y)
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

    y_pred = torch.tensor([1, 2, 3, 4, 5])
    y = torch.tensor([1, 2, 3, 4, 5])
    encoder = LabelEncoder().fit(y)

    print(evaulate_model(y_pred, y, [1, 2, 3], encoder))
