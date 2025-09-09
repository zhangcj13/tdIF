img_h = 416 
img_w = 416
ori_img_h = None
ori_img_w = None
multiscale_range = 5

num_classes = 20

SNN = dict(
    type= 'QUANT',  # 'QUANT'  'STBP'  NONE
    time_step=255,   # Quantized time-step
)

backbone = dict(
    type='DarknetWrapper',
    darknet='darknet7f',
    init_weight=True,
    extract_features=True,
    out_conv=False,
)

featuremap_out_channel = [64, 128, 256, 1024]

# data for box detect
max_boxes = 256
anchors_mask=[[3, 4, 5], [0, 1, 2]]

bbox_heads = dict(type='YoloV3Tiny_Head',
                num_classes=num_classes,
                init_weight=True,
                anchors_mask=anchors_mask,
                in_chns=featuremap_out_channel)

yolo_loss=dict(num_classes=num_classes,
               anchors=[[ 10., 14.],[ 23., 27.],[ 37., 58.],[ 81., 82.],[135., 169.],[344., 319.]],
               input_shape=(int(img_h),int(img_w)),
               anchors_mask=anchors_mask,
               )

data_dir = '/root/siton-data-chenjun/dataset/'
