import torch.nn as nn
import torch

from detection.models.registry import NETS
from ..registry import build_backbones, build_aggregator, build_heads, build_necks,build_head
from snn.quant.utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d,reset_net,replace_activation_by_slip,search_fold_and_remove_bn
from spikingjelly.clock_driven import functional
import copy
from snn.multi_step_layers import replace_ss_by_ms

def accum_mem(o, m):
    for k in m.keys():
        if k in o.keys():
            o[k] += m[k]
        else:
            o[k] = m[k]
    return o

@NETS.register_module
class LaneDetector(nn.Module):
    def __init__(self, cfg):
        super(LaneDetector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbones(cfg)
        self.aggregator = build_aggregator(cfg) if cfg.haskey('aggregator') else None
        self.neck = build_necks(cfg) if cfg.haskey('neck') else None

        self.lane_heads = build_head(cfg.lane_heads, cfg) if cfg.haskey('lane_heads') else None
        
        assert (self.lane_heads is not None), 'No heads defined, please check the config file'

        self.snn_train=False
        self.model_type = 'CNN'
        if cfg.haskey('SNN'):
            self.rebuild_net()
            # self.convert2multistep()
        
        self.eval_snn = False
        self.eval_time_step = 15
        self.multi_step = False

    def rebuild_net(self,):
        params = self.cfg.SNN
        self.time_step = params['time_step']

        if params['type']=='QUANT':
            replace_maxpool = True if 'replace_maxpool' not in params.keys() else params['replace_maxpool']
            if replace_maxpool:
                replace_maxpool2d_by_avgpool2d(self)
            replace_activation_by_floor(self, t=self.time_step)
            self.snn_infer = False
            print(f'>>>>>>>>>>>> quant time-step: {self.time_step} <<<<<<<<<<<<')

        elif params['type']=='SLIP':
            replace_maxpool = True if 'replace_maxpool' not in params.keys() else params['replace_maxpool']
            if replace_maxpool:
                replace_maxpool2d_by_avgpool2d(self)
            replace_activation_by_slip(self, self.time_step, 
                                       params['a'], params['shift1'], params['shift2'], params['a_learnable'])
            self.snn_infer = False

    
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

    def decode(self, output):
        decode_data={}
        decode_data['lane_line'] = self.lane_heads.get_lanes(output)
        return decode_data

    def forward_ann(self, batch):
        output = {}
        fea = self.backbone(batch['img'])

        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)

        if self.training:
            loss_stats={}
            lane_out = self.lane_heads(fea, batch=batch)
            lane_loss = self.lane_heads.loss(lane_out, batch)
            output.update(lane_loss)
            loss_stats.update(lane_loss['loss_stats'])
            output['loss_stats']=loss_stats
        else:
            output.update(self.lane_heads(fea, batch=batch))
        return output

    # def forward_snn(self, batch):

    def forward_snn(self, batch):
        return self.forward_snn_eval(batch)
    
    def forward_snn_eval(self, batch, TS=8):

        assert hasattr(self,'snn_infer'), 'current network can not infer with spiking'
        if not self.snn_infer and not self.snn_train:
            self.convert2snn()
        
        if self.ms:
            if not hasattr(self,'snn_infer_ms'):
                self.convert2multistep()
            return self.forward_snn_ms(batch)
        else:
            return self.forward_snn_ss(batch)
        
    def forward_snn_ms(self, batch):
        reset_net(self)

        encode_x = batch['img']

        encode_x.unsqueeze_(-1)
        x = encode_x.repeat(1,1,1,1, self.eval_time_step)
        
        fea = self.backbone(x)

        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)
        
        ms_out={}

        if self.neuron_type=='tdIF':
            ms_out.update(self.lane_heads.infer_snn(fea, batch=batch))
            rs = [2**(self.eval_time_step-1-t) for t in range(self.eval_time_step)]
            fs = 2**self.eval_time_step-1
            v_sum={}
            for t in range(self.eval_time_step):
                for k in ms_out.keys():
                    if k in v_sum.keys():
                        v_sum[k]+=ms_out[k][...,t]*rs[t]
                    else:
                        v_sum[k]=ms_out[k][...,t]*rs[t]

            for k in v_sum.keys():
                v_sum[k] = v_sum[k]/fs
        else:
            ms_out.update(self.lane_heads.infer_snn(fea, batch=batch))
            
            v_sum={k:torch.mean(ms_out[k], dim=-1, keepdim=False) if k[0]=='m' else ms_out[k]  for k in ms_out.keys()}
            
        output={}
        output.update(self.lane_heads.infer_snn(v_sum, post_process=True,batch=batch))

        return output

    def forward_snn_ss(self, batch):
        reset_net(self)

        encode_x = batch['img']
        v_sum = {}
        # s_sum = {}
        for t in range(self.time_step):
            if encode_x.dim() == 5:
                x = encode_x[t]
            else:
                x = encode_x

            fea = self.backbone(x)

            # s_sum = accum_mem(s_sum,{'f1':fea[-1]})

            if self.aggregator:
                fea[-1] = self.aggregator(fea[-1])

            if self.neck:
                fea = self.neck(fea)

            if self.lane_heads:
                mems = self.lane_heads.infer_snn(fea, batch=batch)
                v_sum = accum_mem(v_sum,mems)
            if self.bbox_heads:
                mems = self.bbox_heads.infer_snn(fea, batch=batch)
                v_sum = accum_mem(v_sum,mems)
            if self.segm_heads:
                mems = self.segm_heads.infer_snn(fea, batch=batch)
                v_sum = accum_mem(v_sum,mems)
        
        # v_sum={k:v_sum[k]/self.time_step for k in v_sum.keys()}
        v_sum={k:v_sum[k]/self.time_step if k[0]=='m' else v_sum[k]  for k in v_sum.keys()}
        
        output={}
        
        if self.lane_heads:
            output.update(self.lane_heads.infer_snn(v_sum, post_process=True,batch=batch))
        if self.bbox_heads:
            output.update(self.bbox_heads.infer_snn(v_sum, post_process=True,batch=batch))
        if self.segm_heads:
            output.update(self.segm_heads.infer_snn(v_sum, post_process=True,batch=batch))

        return output

    def forward(self, batch, forward_snn=False):
            
        if self.eval_snn:
            return self.forward_snn_ms(batch)
        
        return self.forward_ann(batch)
    
