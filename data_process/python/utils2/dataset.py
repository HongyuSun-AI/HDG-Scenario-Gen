import pickle

import numpy as np


import torch
from torch.utils.data import Dataset


class Init_dataset(Dataset):
    def __init__(self, data=None, labels=None):

        self.data = [torch.tensor(item, dtype=torch.float32) for item in data] if data is not None else []
        self.labels = [torch.tensor(label, dtype=torch.int) for label in labels] if labels is not None else []


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        return x, y

    def add_sample(self, data, label):

        self.data.append(data)
        self.labels.append(label)


class Track_dataset(Dataset):
    def __init__(self, data, labels):
        # data: (N, C, ...)

        self.data = torch.tensor(data, dtype=torch.float32)

        self.labels = [(torch.tensor(condition, dtype=torch.float32), torch.tensor(cls, dtype=torch.int)) for condition, cls in labels]

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        x = self.data[idx]
        y = self.labels[idx]
        return x, y

