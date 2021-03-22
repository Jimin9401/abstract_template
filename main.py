import torch
from utils.args import ExperimentArgument


def run(gpu,args):
    args.gpu=gpu
    args.device=gpu


    return NotImplementedError



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    args=ExperimentArgument()

    args.ngpus_per_node=torch.cuda.device_count()
    args.world_size=args.ngpus_per_node

    if args.distributed_training:
        import torch.multiprocessing as mp

        mp.spawn(run, nprocs=args.ngpus_per_node, args=(args,))

    else:
        run("cuda",args)