"""
    lane type to gt_points 
"""
import random
import collections

import albumentations as al
import numpy as np
import copy

from ..registry import PROCESS


@PROCESS.register_module
class ToGTPoints(object):

    def __init__(self, cfg=None):
        self.cfg = cfg

    def __call__(self, data):
        data['gt_points']=copy.deepcopy(data['lane_line'])
        return data



