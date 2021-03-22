from .batch_generator import BatchGen
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
from .logger import logger

def set_init_group(model, args):
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:6557',
                            world_size=args.world_size, timeout=datetime.timedelta(0, 60),
                            rank=args.gpu)

    logger.info("Complete to build process in {} process for Distributed Data Parallel training".format(args.gpu))
    model.to(args.gpu)
    model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)

    return model

class DynamicDistributedSampler(Sampler):
    """
    DistributedSampler For Dynamic Sequence Length dataset

    """
    def __init__(self, batchifer:BatchGen, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.batchfier=batchifer
        self.dataset = batchifer
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(self.batchfier.num_buckets)) ## return 0~ bucket size

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
