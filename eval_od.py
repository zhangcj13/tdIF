#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import sys
import os

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch
from yolox.exp import Exp, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--neuron", type=str, default='tdIF', help="neuron name [A2F, tdID]")
    parser.add_argument("--time_step", type=int, default='8', help="convert snn time step")
    parser.add_argument("--delay", type=int, default='3', help="convert snn time delay")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")#64
    parser.add_argument(
        "-d", "--devices", default=1, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default='configs/yolox_exp/yolov3tiny_voc_quant.py',
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"  # False  True
    )
    # parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False, 
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help="Caching imgs to ram/disk for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,  # False  True
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics. \
        Implemented loggers include `tensorboard` and `wandb`.",
        default="tensorboard"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def load_weight(network,path):
    ckpt_info = torch.load(path)
    print('********** start epoch:', ckpt_info['start_epoch'])
    pretrained_net = ckpt_info['model']
    net_state = network.state_dict()
    state = {}
    for k, v in pretrained_net.items():
        if k not in net_state.keys() or v.size() != net_state[k].size():
            print('skip weights: ' + k)
            continue
        state[k] = v
    network.load_state_dict(state, strict=False)

def eval(args):
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)


    ''' **********************
    get model
    ********************** ''' 
    net=exp.get_model()
    weight_path = args.ckpt    
    load_weight(net, weight_path)

    ''' **********************
    convert model to snn 
    ********************** ''' 
    print('\n\n\n--------------------------------------------------------')
    net.convert2ms(t=args.time_step,
                    neuron=args.neuron,
                    delay=args.delay,
                    )
    net.cuda()
    ''' **********************
    test current mAP
    ********************** ''' 
    evaluator = exp.get_evaluator(batch_size=args.batch_size, is_distributed=False)
    res=evaluator.evaluate(net, False, False, return_outputs=False)
    return res
   

if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    res=eval(args)

    