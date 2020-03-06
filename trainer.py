from abc import abstractmethod, ABC
from overrides import overrides
from itertools import chain
from tqdm import tqdm
from collections import Counter

import math
import numpy as np
import logging

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

logger = logging.getLogger(__file__)


class Trainer(ABC):

    def __init__(self, net, optimizer, tokenizer, args):
        self.net = net
        self.optimizer = optimizer
        self.args = args
        self.tokenizer = tokenizer
        self.PAD_IDX = tokenizer.convert_tokens_to_ids("<pad>")

        self.criterion = nn.CrossEntropyLoss()

        self.END_IDX = tokenizer.convert_tokens_to_ids("<eos>")

    @abstractmethod
    def train(self, batchfier):
        return NotImplementedError

    @abstractmethod
    def eval(self, batchfier):
        return NotImplementedError