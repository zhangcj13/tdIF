import torch
from torch import nn
import torch.nn.functional as F

from mmcv.cnn.bricks.registry import ACTIVATION_LAYERS

@ACTIVATION_LAYERS.register_module()
class MyRelu(nn.Module):
    def __init__(self, max_value=None):
        super(MyRelu, self).__init__()
        self.max_value=max_value

    def forward(self, x):
        output = torch.maximum(x,torch.zeros_like(x))
        if self.max_value is not None:
            output=torch.minimum(output,torch.ones_like(x)*self.max_value)
        return output



