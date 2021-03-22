from tqdm import tqdm
import torch
from torch.utils.data import IterableDataset,DataLoader
import random
import os
from .logger import Logger

class TrainerBase:
    def __init__(self,args, model, train_batchfier, test_batchfier, optimizers, schedulers,
                 update_step, criteria, clip_norm, mixed_precision,num_class:int=2):
        self.args=args
        self.model = model
        self.train_batchfier = train_batchfier
        self.test_batchfier = test_batchfier
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.criteria = criteria
        self.step = 0
        self.update_step = update_step
        self.mixed_precision = mixed_precision
        self.clip_norm = clip_norm
        self.num_class = num_class

        self.__init_writer()

    def __init_writer(self):

        postfix = "transformer" if self.args.scratch else "bert"

        board_directory=os.path.join(self.args.vis_dir,self.args.savename,postfix)
        if not os.path.isdir(board_directory):
            os.makedirs(board_directory)
        self.writer=Logger(log_dir=board_directory)


    def reformat_inp(self, inp):
        raise NotImplementedError

    def train_epoch(self):

        model = self.model
        batchfier = self.train_batchfier
        criteria = self.criteria
        optimizer = self.optimizers
        scheduler = self.schedulers

        if isinstance(batchfier, IterableDataset):

            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.batch_size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate,pin_memory=True)

        # cached_data_loader=get_cached_data_loader(batchfier,batchfier.size,custom_collate=batchfier.collate,shuffle=False)
        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0

        model.zero_grad()
        pbar = tqdm(batchfier,total=len(batchfier.dataset))

        return NotImplementedError

    def test_epoch(self):
        model = self.model
        batchfier = self.test_batchfier


        return NotImplementedError

    def update_description(self, description, n_bar):
        description += 'lr : %f, iter : %d ' %(self.schedulers.get_last_lr()[0], n_bar)
        return description
