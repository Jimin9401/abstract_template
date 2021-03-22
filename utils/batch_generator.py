from torch.utils.data import IterableDataset
import pandas as pd


class BatchGenerator(IterableDataset):
    def __init__(self,df:pd.DataFrame,batch_size:int=16,padding_index:int=0,max_length=512):
        self.df=df
        # self.lens=dataset.shape[0]
        self.batch_size=batch_size
        self.padding_index=padding_index
        self.max_length=max_length

    def collate(self,input):
        """
        :param input:
        :return:
        """
        return NotImplementedError


    def __iter__(self):

        return NotImplementedError
