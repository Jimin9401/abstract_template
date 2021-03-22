
import torch.nn as nn
from transformers import AutoConfig,AutoModel


class BaseLine(nn.Module):

    def __init__(self,args,n_class,n_intermediate=2):
        super(BaseLine, self).__init__()
        self.args=args
        self.config = AutoConfig.from_pretrained(args.encoder_class)
        self.transformer=AutoModel.from_pretrained(args.encoder_class,config=self.config)
        self.__init_classifier_weight()

    def __init_classifier_weight(self):
        nn.init.xavier_normal_(self.dense.weight.data)
        nn.init.xavier_normal_(self.classifier.weight.data)

    def forward(self,inps,attention_mask):
        """
        inps (N,seq_lens)
        attention_mask (N,seq_lens)
        return (N, hidden_size)
        """

        return NotImplementedError
