# encoding: utf-8
import sys
import os

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as BaseExp

import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from detection.models.registry import build_backbones, build_aggregator, build_heads, build_necks,build_head
from snn.quant.utils import replace_relu_by_leakyrelu
from snn.quant.utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d,reset_net,replace_activation_by_slip,search_fold_and_remove_bn
from snn.multi_step_layers import replace_ss_by_ms

class YOLOV3(nn.Module):
    def __init__(self, cfg):
        super(YOLOV3, self).__init__()
        
        self.cfg = cfg
        self.backbone = build_backbones(cfg)
        self.neck = build_necks(cfg) if cfg.haskey('neck') else None
        self.head = build_head(cfg.bbox_heads, cfg)

        self.eval_snn = False
        self.eval_time_step = 15
        self.multi_step = False
    
    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        if self.eval_snn:
            return self.forward_snn(x)
        fea = self.backbone(x)

        if self.neck:
            fea = self.neck(fea)

        if self.training:
            assert targets is not None
            batch={'img':x,'bounding_box':self.target2yolov3(targets,x)}
            bbox_out = self.head(fea, batch=batch)
            bbox_loss = self.head.loss(bbox_out, batch)
            outputs = {
                "total_loss": bbox_loss['bbox_loss'],
            }
        else:
            outputs = self.head(fea)
            decode_bbox = self.head.get_bboxes(outputs, sep_score=True)
            outputs=self.postprocess(decode_bbox)

        return outputs
    
    def convert2snn(self,t:int=15):
        replace_activation_by_neuron(self)
        # print('net: ',self)
        self.eval_snn = True
        self.eval_time_step = t
    
    def convert2ms(self,t:int=15,neuron='IF',delay=None):
        if not  self.eval_snn:
            self.convert2snn(t)
        else:
            self.eval_time_step = t
        
        replace_ss_by_ms(self,neuron=neuron,delay=delay)
        self.multi_step = True
        self.neuron_type=neuron

    def forward_snn(self, encode_x):
        if self.multi_step:
            return self.forward_snn_ms(encode_x)
        return self.forward_snn_ss(encode_x)

    def forward_snn_ss(self, encode_x):
        reset_net(self)

        outputs = None
        for t in range(self.eval_time_step):
            if encode_x.dim() == 5:
                x = encode_x[t]
            else:
                x = encode_x 
        
            fea = self.backbone(x)

            if self.neck:
                fea = self.neck(fea)

            mem = self.head(fea)

            if t==0:
                outputs = mem
            else:
                outputs['pred_boxes'][0]+=mem['pred_boxes'][0]
                outputs['pred_boxes'][1]+=mem['pred_boxes'][1]

        outputs['pred_boxes'][0]=outputs['pred_boxes'][0]/self.eval_time_step
        outputs['pred_boxes'][1]=outputs['pred_boxes'][1]/self.eval_time_step


        decode_bbox = self.head.get_bboxes(outputs, sep_score=True)
        outputs=self.postprocess(decode_bbox)

        return outputs
    
    def forward_snn_ms(self, encode_x):
        reset_net(self)

        encode_x.unsqueeze_(-1)
        x = encode_x.repeat(1,1,1,1, self.eval_time_step)

        fea = self.backbone(x)

        if self.neck:
            fea = self.neck(fea)

        outputs = self.head(fea)
        if self.neuron_type=='tdIF':
            rs = [2**(self.eval_time_step-1-t) for t in range(self.eval_time_step)]
            fs = 2**self.eval_time_step-1
            o1=0
            o2=0
            o3=0
            for t in range(self.eval_time_step):
                o1 += outputs['pred_boxes'][0][...,t]*rs[t]
                o2 += outputs['pred_boxes'][1][...,t]*rs[t]
                o3 += outputs['pred_boxes'][2][...,t]*rs[t]
            outputs['pred_boxes'][0] = o1/fs
            outputs['pred_boxes'][1] = o2/fs
            outputs['pred_boxes'][2] = o3/fs
        else:
            outputs['pred_boxes'][0] = torch.mean(outputs['pred_boxes'][0], dim=-1, keepdim=False)
            outputs['pred_boxes'][1] = torch.mean(outputs['pred_boxes'][1], dim=-1, keepdim=False)
            outputs['pred_boxes'][2] = torch.mean(outputs['pred_boxes'][2], dim=-1, keepdim=False)

        
        decode_bbox = self.head.get_bboxes(outputs, sep_score=True)
        outputs=self.postprocess(decode_bbox)

        return outputs
    def visualize(self, x, targets, save_prefix="assign_vis_"):
        # fpn_outs = self.backbone(x)
        # self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
        return

    def target2yolov3(self,targets,x):
        _,_,H,W = x.shape
        temp = torch.cat((targets[...,1:],targets[...,:1]),dim=2)
        temp[...,0:4:2]*=1.0/W
        temp[...,1:4:2]*=1.0/H
        return temp
    
    def postprocess(self,batch_pred):
        post_pred=[]
        for pred in batch_pred:
            boxes=[]
            for box in pred:
                # t,l,b,r=box['box']
                # boxes.append([l,t,r,b])
                tbox = box['box'].copy()
                tbox.append(box['o_score'])
                tbox.append(box['c_score'])
                tbox.append(box['class'])
                boxes.append(tbox)

            boxes = torch.from_numpy(np.array(boxes).astype(np.float32))
            post_pred.append(boxes)
        
        return post_pred
            


class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 80
        self.depth = 0.33
        self.width = 0.50
        self.warmup_epochs = 1

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
        # To disable multiscale training, set the value to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir

        # name of annotation file for training
        self.train_ann = "instances_train2017.json"
        # name of annotation file for evaluation
        self.val_ann = "instances_val2017.json"
        # name of annotation file for testing
        self.test_ann = "instances_test2017.json"

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 5
        # max training epoch
        self.max_epoch = 50 #200 50 90
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 64.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs =30 #50 30 40
        # apply EMA during training
        self.ema = True

        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.test_size = (640, 640)

        self.data_dir = './dataset/COCO/'
    
    def get_model(self):
        from detection.utils.config import Config
        import argparse

        cfg = Config.fromfile('configs/COCO/resnet34_yolov3head.py')
        self.model = YOLOV3(cfg)

        params = cfg.SNN
        self.time_step = params['time_step']
        if params['type']=='QUANT':
            replace_maxpool2d_by_avgpool2d(self.model)
            replace_activation_by_floor(self.model, t=self.time_step)
            print(f'********** training time-step:{self.time_step} **********')

        self.model.train()
        return self.model

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import VOCDetection, TrainTransform

        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)
        # legacy = True

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="val2017" if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
            need_postprocess=False,
        )

    
