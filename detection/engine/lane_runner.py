import time
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import cv2

from detection.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from detection.datasets import build_dataloader
from detection.utils.recorder import build_recorder
from detection.utils.net_utils import save_model, load_network,load_network_specified
from mmcv.parallel import MMDataParallel 
import os
from yolox.utils.model_utils import get_model_info
import time

from .runner import Runner

class LaneDetectorRunner(Runner):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Model_Summary = get_model_info(self.net.module,(800,320),
        #                                input={'img':torch.zeros((1, 3, 32, 32), device=next(self.net.parameters()).device)})
        
        # self.recorder.logger.info(f'Model Summary: {Model_Summary}')
    
    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        if self.cfg.finetune_from:
            load_network_specified(self.net, self.cfg.finetune_from,
                                  logger=self.recorder.logger)
            return

        load_network_specified(self.net, self.cfg.load_from,
                               logger=self.recorder.logger)

    def get_total_loss(self,output):
        lane_loss = output['loss'] if 'loss' in output.keys() else 0
        return lane_loss
    
    def train(self):
        if self.cfg.haskey('SNN'):
            if self.cfg.SNN['type']=='QUANT':
               self.recorder.logger.info(f'----->>>>> QUANT training step:{self.cfg.SNN.time_step} <<<<<-----')
        self.recorder.logger.info('Build train loader...')
        train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True, drop_last=True)

        self.recorder.logger.info('Start training...')
        for epoch in range(self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, train_loader)
            if (epoch + 1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt()
            if (epoch + 1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate()
            if self.recorder.step >= self.cfg.total_iter:
                break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()

    def train_epoch(self, epoch, train_loader):
        self.net.train()

        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            data = self.to_cuda(data)
            output = self.net(data)
            self.optimizer.zero_grad()
            loss = self.get_total_loss(output)
            loss.backward()
            # loss.backward(retain_graph = True)
            self.optimizer.step()
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            if self.warmup_scheduler:
                self.warmup_scheduler.dampen()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')
                                        
    def validate(self, return_metric=False):
        from detection.utils.torch_utils import fuse_net
        from detection.utils.quantity import quantify_net
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val, self.cfg, is_train=False,drop_last=False)
        self.net.eval()

        # # fuse_net(self.net)
        # quantify_net(self.net)
        # print(self.net)

        predictions = {}
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                # output = self.net(data)
                output = self.net(data, forward_snn=False) # False True
                output = self.net.module.decode(output)

                for k in output.keys():
                    if k not in predictions.keys():
                        predictions[k]=[]
                    predictions[k].extend(output[k])

            if self.cfg.view:
                self.val_loader.dataset.view(output, data['meta'])

        out = self.val_loader.dataset.evaluate(predictions, self.cfg.work_dir,return_metric=return_metric)

        if return_metric:
            return out

        self.recorder.logger.info(out)
        metric = np.sum([out[k] if 'log' not in k else 0 for k in out.keys()])
        if metric > self.metric:
            self.metric = metric
            self.save_ckpt(is_best=True)
        self.recorder.logger.info('Best metric: ' + str(self.metric))
    
    def visual_quality(self, image_ids=[], save_dir='./'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        from detection.utils.torch_utils import fuse_net
        from detection.utils.quantity import quantify_net
        
        self.cfg.batch_size=1
        
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val, self.cfg, is_train=False,drop_last=False)
        self.net.eval()

        predictions = {}
        vq_cnt = len(image_ids)
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            if i not in image_ids:
                continue
            if vq_cnt<=0:
                break
            vq_cnt-=1
            frame = cv2.imread(data['meta'].data[0][0]['full_img_path'])
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data, forward_snn=False) # False True
                output = self.net.module.decode(output)

            lanes=output['lane_line'][0] 

            colors=[(0, 0, 255),(0, 255, 0),(255, 0, 0),(0, 255, 255),(255, 0, 255),(255, 255, 0)]
            for nl, l in enumerate(lanes):
                color=colors[nl] if nl <6 else (255,255,255)
                points = l.points
                points[:, 0] *= frame.shape[1]
                points[:, 1] *= frame.shape[0]
                points = points.round().astype(int)
                # xs, ys = points[:, 0], points[:, 1]
                for curr_p, next_p in zip(points[:-1], points[1:]):
                    frame = cv2.line(frame,
                                    tuple(curr_p),
                                    tuple(next_p),
                                    color=color,
                                    thickness=9 )
            
            img_save_path = os.path.join(save_dir,str(i)+'.jpg')

            cv2.imwrite(img_save_path,frame)

        return None
