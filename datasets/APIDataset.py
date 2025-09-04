import torch
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset


class APIDataset(Dataset):
    def __init__(
        self,
        data: ndarray,
        normal_key_api_sequence: ndarray,
        abnormal_key_api_sequence: ndarray,
        label: ndarray,
        *,
        dtype=None
    ):
        self.x: Tensor = torch.tensor(data, dtype=torch.long)
        self.normal_key_api_sequence: Tensor = torch.tensor(normal_key_api_sequence, dtype=torch.long)
        self.abnormal_key_api_sequence: Tensor = torch.tensor(abnormal_key_api_sequence, dtype=torch.long)
        self.y: Tensor = torch.tensor(label, dtype=dtype)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.normal_key_api_sequence[idx], self.abnormal_key_api_sequence[idx], self.y[idx]
