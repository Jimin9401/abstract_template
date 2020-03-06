import numpy as np
import random
import pandas as pd

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, DataLoader, Dataset
from abc import abstractmethod


class ABCBatchFier(IterableDataset):

    def __init__(self, data: dict, tokenizer, padding_idx=0, device="cuda"):
        super(BaseBatchFier, self).__init__()
        self.data = data
        self.padding_idx = padding_idx
        self._tokenizer = tokenizer

        # self.bucketing()

    def __len__(self):
        return len(self.data)

    @abstractmethod
    def iterator(self):
        return NotImplementedError

    @abstractmethod
    def __iter__(self):
        return self.iterator()

    @abstractmethod
    def collate(self, batch):
        return NotImplementedError


