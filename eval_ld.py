import sys
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
from detection.utils.net_utils import save_model, load_network,load_network_specified


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--work_name', default = 'resnet18_condlane', help='name for save')
    parser.add_argument('--config', default = f'configs/Tusimple/resnet18_condlane.py', help='train config file path')
    
    parser.add_argument("--neuron", type=str, default='tdIF', help="neuron name [A2F, tdID]")
    parser.add_argument("--time_step", type=int, default='8', help="convert snn time step")
    parser.add_argument("--delay", type=int, default='3', help="convert snn time delay")

    
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
        default=True,
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    return args

def eval():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)

    cfg.load_from = args.load_from
    cfg.finetune_from = args.finetune_from
    cfg.view = args.view
    cfg.seed = args.seed

    cfg.work_dirs = args.work_dirs + '/' + cfg.dataset.train.type + '/' +args.work_name  +'_eval'
    
    cudnn.benchmark = True
    # cudnn.fastest = True

    runner = LaneDetectorRunner(cfg)

    weight_path=args.load_from

    load_network_specified(runner.net, weight_path, logger=runner.recorder.logger)
    
    runner.net.module.convert2ms(t=args.time_step,
                    neuron=args.neuron,
                    delay=args.delay,
                    )
    runner.net.cuda()
    print(runner.net)
    return runner.validate(return_metric=True)

        
if __name__ == '__main__':
    res=eval()
    print(res)
   


