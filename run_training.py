import argparse
import os

import numpy as np
from train.trainer import Trainer
from utils.base_utils import load_cfg
import torch
import torch.utils.data.distributed
import torch.distributed as dist

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/train/gen/neuray_gen_depth_train.yaml')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--distributed', action='store_true', help='if use distributed training')

    args = parser.parse_args()
    print(args.distributed)
    print(args.local_rank)
    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
        
        cfg = load_cfg(args.cfg)
        cfg["total_step"] = cfg["total_step"] / ngpus_per_node
        cfg["lr_cfg"]["decay_step"] = cfg["lr_cfg"]["decay_step"] / ngpus_per_node
        print('test')
        trainer = Trainer(cfg, exp_name=args.exp_name, distributed=True, local_rank=args.local_rank)
        trainer.run()

    else:
        trainer = Trainer(load_cfg(args.cfg), exp_name=args.exp_name, distributed=False)
        trainer.run()