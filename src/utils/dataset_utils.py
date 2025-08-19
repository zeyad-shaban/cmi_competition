import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset


class SensorDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor, should_augment=False):
        super().__init__()
        self.data = features
        self.targets = targets
        self.should_augment = should_augment

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]

        if self.should_augment:
            raise NotImplementedError("Augmentations not implemented")

        return x, y

    def __len__(self):
        return len(self.data)


def sensor_collate_train_fn(batch, n_imu, n_classes=18, mixup_alpha=0.4):
    x_batch = torch.stack([item[0] for item in batch])
    y_batch = torch.stack([item[1] for item in batch])

    # ---- Mixup Augmentation ----
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    shuffled_indices = torch.randperm(x_batch.shape[0])

    x2 = x_batch[shuffled_indices]
    x_mixed = lam * x_batch + (1 - lam) * x2

    y1 = nn.functional.one_hot(y_batch, num_classes=n_classes)
    y2 = y1[shuffled_indices]
    y_mixed = lam * y1 + (1 - lam) * y2
    # ---- End Mixup ----

    # ---- Randomly disable non IMU ----
    shuffled_indices = torch.randperm(x_batch.shape[0] // 2)
    x_mixed[shuffled_indices, n_imu:, :] = -1
    # ---- End Randomly disable non IMU ----

    return x_mixed, y_mixed


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dummy_input = torch.randn([64, 20, 100])
    dummy_targets = torch.randint(0, 5, [64])

    dataset = SensorDataset(dummy_input, dummy_targets)
    dataloder = DataLoader(dataset, 8, shuffle=True, collate_fn=lambda batch: sensor_collate_train_fn(batch, n_imu=5))
    for x, y in dataloder:
        print(y.shape)
