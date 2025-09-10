import torch
import torch.nn as nn
import numpy as np
from ..backbones.darknet import Conv,_initialize_weights
from ..losses.yolo_loss import YOLOLoss
from detection.utils.utils_bbox import DecodeBox
from collections import OrderedDict
from ..registry import HEADS

@HEADS.register_module
class YoloV3Tiny_Head(nn.Module):
    def __init__(self, num_classes: int = 80,
                 init_weight: bool = True,
                 anchors_mask=[[3, 4, 5], [0, 1, 2]],
                 in_chns=[256,1024],
                 neck=True,
                 cfg=None) -> None:
        super().__init__()
        self.cfg=cfg
        if init_weight:
            self.apply(_initialize_weights)

        self.neck = nn.Sequential(
            Conv(in_channels=in_chns[-1], out_channels=256, kernel_size=1),
        ) if neck else None
        self.head1 = nn.Sequential(
            Conv(in_channels=256 if neck else in_chns[-1], out_channels=512, kernel_size=3),
            nn.Conv2d(in_channels=512,out_channels=len(anchors_mask[0]) * (num_classes + 5),kernel_size=1,
                stride=1,bias=True,)
        )
        self.branch = nn.Sequential(
            Conv(in_channels=256, out_channels=128, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.head2 = nn.Sequential(
            Conv(in_channels=in_chns[-2]+128, out_channels=256, kernel_size=3),
            nn.Conv2d(in_channels=256, out_channels=len(anchors_mask[1]) * (num_classes + 5), kernel_size=1,
                      stride=1, bias=True, )
        )

    def forward(self, features,**kwargs):
        assert len(features)>=2
        f1,f2 = features[-2:]

        if self.neck:
            x = self.neck(f2)
        else:
            x=f2
        out1 = self.head1(x)

        x = self.branch(x)
        x = torch.cat([f1, x], dim=1)
        out2 = self.head2(x)

        output = {'pred_boxes': [out1, out2]}

        return output
    
    def infer_snn(self, inputs, post_process=False,**kwargs):
        if not post_process:
            assert len(inputs)>=2
            f1,f2 = inputs[-2:]

            if self.neck:
                x = self.neck(f2)
            else:
                x=f2
            out1 = self.head1(x)

            x = self.branch(x)
            x = torch.cat([f1, x], dim=1)
            out2 = self.head2(x)
            output= {'m_out1': out1, 'm_out2': out2}
        else:
            assert 'm_out1' in inputs.keys() and 'm_out2' in inputs.keys()
            output = {'pred_boxes': [inputs['m_out1'], inputs['m_out2']]}
        return output

    def loss(self,output, batch):
        params = self.cfg.yolo_loss

        pred_boxes = output['pred_boxes']
        targets = batch['bounding_box']

        img_shape = batch['img'].shape[2:4]

        yolo_loss = YOLOLoss(cuda=pred_boxes[0].is_cuda, **params)
        loss_value_all = 0
        num_pos_all = 0

        for l in range(len(pred_boxes)):
            loss_item, num_pos = yolo_loss(l, pred_boxes[l], targets,img_shape=img_shape)
            loss_value_all += loss_item
            num_pos_all += num_pos
        boxes_loss = loss_value_all / num_pos_all

        return {'bbox_loss': boxes_loss, 'loss_stats': {'bbox_loss': boxes_loss}}

    def get_bboxes(self,output,sep_score=False):
        pred_boxes = output['pred_boxes']
        params = self.cfg.yolo_loss
        bbox_util = DecodeBox(**params)
        if self.cfg.ori_img_h is None or self.cfg.ori_img_w is None:
            init_image_shape = np.array([self.cfg.img_h, self.cfg.img_w])
        else:
            init_image_shape = np.array([self.cfg.ori_img_h, self.cfg.ori_img_w])

        with torch.no_grad():
            outputs = bbox_util.decode_box(pred_boxes)
            results = bbox_util.non_max_suppression(torch.cat(outputs, 1), bbox_util.num_classes,
                                                    bbox_util.input_shape,
                                                    init_image_shape, False, 
                                                    conf_thres=0.001, nms_thres=0.5,
                                                    # conf_thres=0.5, nms_thres=0.3,
                                                    )
        batch_boxes = []
        for result in results:
            if result is None:
                batch_boxes.append([])
                continue

            top_label = np.array(result[:, 6], dtype='int32')
            top_conf = result[:, 4] * result[:, 5]
            top_boxes = result[:, :4]

            if sep_score:
                obj_score = result[:, 4]
                cls_score = result[:, 4]

            max_boxes = 100
            top_100 = np.argsort(top_conf)[::-1][:max_boxes]
            top_boxes = top_boxes[top_100]
            top_conf = top_conf[top_100]
            top_label = top_label[top_100]
            if sep_score:
                obj_score = obj_score[top_100]
                cls_score = cls_score[top_100]

            boxes = []
            for i, c in list(enumerate(top_label)):
                predicted_class = int(c)
                box = top_boxes[i]
                score = top_conf[i]

                top, left, bottom, right = box
                if c >=  bbox_util.num_classes:
                    continue
                # boxes.append({'box': [top, left, bottom, right], 'class': predicted_class, 'score': score})
                
                if sep_score:
                    boxes.append({'box': [left, top, right, bottom], 'class': predicted_class, 
                                  'score': score,'o_score':obj_score[i],'c_score':cls_score[i]})
                else:
                    boxes.append({'box': [left, top, right, bottom], 'class': predicted_class, 'score': score,})
            batch_boxes.append(boxes)

        return batch_boxes


def conv2d(filter_in, filter_out, kernel_size,inplace=True):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        # ("relu", nn.LeakyReLU(0.1)),
        ("relu", nn.ReLU(inplace=inplace)),
    ]))

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m

@HEADS.register_module
class YoloV3_Head(nn.Module):
    def __init__(self, num_classes: int = 80,
                 init_weight: bool = True,
                 anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 in_chns = [256,512,1024],
                 cfg=None) -> None:
        super().__init__()
        self.cfg=cfg
        if init_weight:
            self.apply(_initialize_weights)

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #------------------------------------------------------------------------#
        self.last_layer0            = make_last_layers([512, 1024], in_chns[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv       = conv2d(512, 256, 1)
        self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1            = make_last_layers([256, 512], in_chns[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer2_conv       = conv2d(256, 128, 1)
        self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2            = make_last_layers([128, 256], in_chns[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, features,**kwargs):
        assert len(features)>=3
        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        x2, x1, x0 = features[-3:]

        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        #---------------------------------------------------#
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out0_branch = self.last_layer0[:5](x0)
        out0        = self.last_layer0[5:](out0_branch)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layer1[:5](x1_in)
        out1        = self.last_layer1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)
        #---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2 = self.last_layer2(x2_in)

        output = {'pred_boxes': [out0, out1, out2]}

        return output
    
    def infer_snn(self, inputs, post_process=False,**kwargs):
        if not post_process:
            assert len(inputs)>=3
            x2, x1, x0 = inputs[-3:]

            out0_branch = self.last_layer0[:5](x0)
            out0        = self.last_layer0[5:](out0_branch)

            x1_in = self.last_layer1_conv(out0_branch)
            x1_in = self.last_layer1_upsample(x1_in)

            x1_in = torch.cat([x1_in, x1], 1)

            out1_branch = self.last_layer1[:5](x1_in)
            out1        = self.last_layer1[5:](out1_branch)

            x2_in = self.last_layer2_conv(out1_branch)
            x2_in = self.last_layer2_upsample(x2_in)

            x2_in = torch.cat([x2_in, x2], 1)

            out2 = self.last_layer2(x2_in)

            output= {'m_out0': out0, 'm_out1': out1, 'm_out2': out2}
        else:
            assert 'm_out0' in inputs.keys() and 'm_out1' in inputs.keys() and 'm_out2' in inputs.keys()
            output = {'pred_boxes': [inputs['m_out0'], inputs['m_out1'], inputs['m_out2']]}
        return output

    def loss(self,output, batch):
        params = self.cfg.yolo_loss

        pred_boxes = output['pred_boxes']
        targets = batch['bounding_box']

        img_shape = batch['img'].shape[2:4]

        yolo_loss = YOLOLoss(cuda=pred_boxes[0].is_cuda, **params)
        loss_value_all = 0
        num_pos_all = 0

        for l in range(len(pred_boxes)):
            loss_item, num_pos = yolo_loss(l, pred_boxes[l], targets,img_shape=img_shape)
            loss_value_all += loss_item
            num_pos_all += num_pos
        boxes_loss = loss_value_all / num_pos_all

        return {'bbox_loss': boxes_loss, 'loss_stats': {'bbox_loss': boxes_loss}}

    def get_bboxes(self,output,sep_score=False):
        pred_boxes = output['pred_boxes']
        params = self.cfg.yolo_loss
        bbox_util = DecodeBox(**params)
        if self.cfg.ori_img_h is None or self.cfg.ori_img_w is None:
            init_image_shape = np.array([self.cfg.img_h, self.cfg.img_w])
        else:
            init_image_shape = np.array([self.cfg.ori_img_h, self.cfg.ori_img_w])

        with torch.no_grad():
            outputs = bbox_util.decode_box(pred_boxes)
            results = bbox_util.non_max_suppression(torch.cat(outputs, 1), bbox_util.num_classes,
                                                    bbox_util.input_shape,
                                                    init_image_shape, False, 
                                                    conf_thres=0.001, nms_thres=0.5,
                                                    # conf_thres=0.5, nms_thres=0.3,
                                                    )
        batch_boxes = []
        for result in results:
            if result is None:
                batch_boxes.append([])
                continue

            top_label = np.array(result[:, 6], dtype='int32')
            top_conf = result[:, 4] * result[:, 5]
            top_boxes = result[:, :4]

            if sep_score:
                obj_score = result[:, 4]
                cls_score = result[:, 4]

            max_boxes = 100
            top_100 = np.argsort(top_conf)[::-1][:max_boxes]
            top_boxes = top_boxes[top_100]
            top_conf = top_conf[top_100]
            top_label = top_label[top_100]
            if sep_score:
                obj_score = obj_score[top_100]
                cls_score = cls_score[top_100]

            boxes = []
            for i, c in list(enumerate(top_label)):
                predicted_class = int(c)
                box = top_boxes[i]
                score = top_conf[i]

                top, left, bottom, right = box
                if c >=  bbox_util.num_classes:
                    continue
                # boxes.append({'box': [top, left, bottom, right], 'class': predicted_class, 'score': score})
                
                if sep_score:
                    boxes.append({'box': [left, top, right, bottom], 'class': predicted_class, 
                                  'score': score,'o_score':obj_score[i],'c_score':cls_score[i]})
                else:
                    boxes.append({'box': [left, top, right, bottom], 'class': predicted_class, 'score': score,})
            batch_boxes.append(boxes)

        return batch_boxes
