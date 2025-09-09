img_h = 416 
img_w = 416
ori_img_h = None
ori_img_w = None
multiscale_range = 5

num_classes = 20


SNN = dict(
    type= 'QUANT', 
    time_step=255,
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet34',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

featuremap_out_channel = [64, 128, 256, 512]

# data for box detect
max_boxes = 256
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

bbox_heads = dict(type='YoloV3_Head',
                num_classes=num_classes,
                init_weight=True,
                anchors_mask=anchors_mask,
                in_chns=featuremap_out_channel)

yolo_loss=dict(num_classes=num_classes,
               anchors=[[ 10., 13.],[ 16., 30.],[ 33., 23.],[ 30., 61.],[62., 45.],[59., 119.], [ 116., 90.],[156., 198.],[373., 326.]],
               input_shape=(int(img_h),int(img_w)),
               anchors_mask=anchors_mask,
               )
