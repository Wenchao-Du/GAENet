#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2021/09/25
@Author  :   Garified Du
@Version :   1.0
@License :   Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
@Desc    :   if anything about the descriptions, please put them here. else None
'''

# here put the import lib
import os
import sys
import argparse

from trainer import train
from Utils.utils import get_config
import torch.multiprocessing as mp
#sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description='Pytorch Dist training for DC')
parser.add_argument('--config', type=str, default='configs/kitti.yaml')
parser.add_argument('--local_rank',
                    type=int,
                    default=-1,
                    help='node rank for distributed training!')
args = parser.parse_args()
config = get_config(args.config)
os.environ["CUDA_VISIBLE_DEVICES"] = config["gpuid"]
os.environ['MASTER_ADDR'] = 'localhost'
os.environ["MASTER_PORT"] = config['port']


def main():
    gpu_num = len(config['gpuid'].split(','))
    config['gpus'] = gpu_num
    if not config['eval']:
        if gpu_num == 1:
            train(config, 0)
        else:

            spawn_context = mp.spawn(train,
                                     nprocs=gpu_num,
                                     args=(config, ),
                                     join=True)
            while not spawn_context.join():
                pass
            for process in spawn_context.processes:
                if process.is_alive():
                    process.terminate()
                process.join()


if __name__ == "__main__":
    main()
