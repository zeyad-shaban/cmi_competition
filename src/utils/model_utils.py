import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import torch
import pandas as pd
from torch.utils.data import DataLoader 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaulate_model(model: nn.Module, X_test: torch.Tensor, y_test, target_gestures_encoded: torch.Tensor, encoder: LabelEncoder):
    model.eval()  # not sure if it's best practice to have this here or expect the model to come in eval mode
    y_pred = torch.argmax(model(X_test), dim=1)

    conf_matrix_result = confusion_matrix(y_test, y_pred)
    clsf_report_result = pd.DataFrame(classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True, zero_division=0)).T

    y_test_binary = torch.isin(y_test, target_gestures_encoded)
    y_pred_binary = torch.isin(y_pred, target_gestures_encoded)
    f1_binary = f1_score(y_test_binary, y_pred_binary, average="binary")

    f1_macro = f1_score(y_test, y_pred, average="macro")
    competition_evaluation = 0.5 * f1_binary + 0.5 * f1_macro

    return {
        "confusion_matrix": conf_matrix_result,
        "classification_report": clsf_report_result,
        "f1_binary": f1_binary,
        "f1_macro": f1_macro,
        "competition_evaluation": competition_evaluation,
    }

def train_model(model: nn.Module, dataloader: DataLoader, n_epochs: int, should_log=True):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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
        if (epoch) % 20 == 0 and should_log:
            print(f"{epoch} - loss_avg: {loss_avg}")


if __name__ == '__main__':
    from model import SimpleModel
    n_classes = 6
    dummy_features = torch.randn((64, n_classes), dtype=torch.float32)
    dummy_target = torch.randint(0, n_classes, (64,), dtype=torch.long)
    dummy_encoder = LabelEncoder()
    
    dummy_encoder.fit(dummy_target.numpy())
    target_gestures_encoded = torch.arange(0, n_classes)
    print(target_gestures_encoded)
    
    model = SimpleModel(n_classes)
    
    print(evaulate_model(model, dummy_features, dummy_target, target_gestures_encoded, dummy_encoder))