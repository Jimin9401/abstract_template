from torch.utils.data import dataset, dataloader
import torch
from copy import deepcopy


class DatasetBase(dataset.Dataset):
    def __init__(self, features, tokenizer):
        self.features = features
        self.tokenizer = tokenizer
    def __getitem__(self, idx):
        feature = deepcopy(self.features[idx])
        feature.text = self.preprocess(feature.context)

        return feature

    def preprocess(self,inps):

        return NotImplementedError

    def __len__(self):
        return len(self.features)

