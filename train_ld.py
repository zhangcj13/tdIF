import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import random
from detection.utils.config import Config
from detection.engine.lane_runner import LaneDetectorRunner
from detection.datasets import build_dataloader

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from detection.models.registry import build_backbones, build_aggregator, build_heads, build_necks,build_head,build_separate_backbones,build_separate_necks
from spikingjelly.clock_driven import functional
from einops import rearrange, repeat
from snn.quant.utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d,reset_net,replace_activation_by_slip,search_fold_and_remove_bn
from snn.multi_step_layers import replace_ss_by_ms



def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)

    cfg.load_from = args.load_from
    cfg.finetune_from = args.finetune_from
    cfg.view = args.view
    cfg.seed = args.seed
    
    cfg.work_dirs = args.work_dirs + '/' + cfg.dataset.train.type + '/' +args.work_name
    if cfg.haskey('SNN'):
        params = cfg.SNN
        if params['type']=='QUANT':
            quant_ts = params['time_step']
            cfg.work_dirs = cfg.work_dirs+f'_ts{quant_ts}'

    cudnn.benchmark = True
    # cudnn.fastest = True

    runner = LaneDetectorRunner(cfg)

    if args.validate:
        runner.validate()
    else:
        runner.train()

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--work_name', default = 'resnet18_condlane', help='name for save')
    parser.add_argument('--config', default = f'configs/Tusimple/resnet18_condlane.py', help='train config file path')
    parser.add_argument(
        '--work_dirs', type=str, default='work_dirs',
        help='work dirs')
    parser.add_argument(
        '--load_from', default=None,
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--finetune_from', default=None,
        help='whether to finetune from the checkpoint')
    parser.add_argument(
        '--view', action='store_true', 
        help='whether to view')
    parser.add_argument(
        '--validate',
        default=False,
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()


