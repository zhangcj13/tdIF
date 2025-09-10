import random
import cv2
import numpy as np
import torch
import numbers
import collections
from PIL import Image
from mmcv.parallel import DataContainer as DC
from einops import rearrange, repeat
from ..registry import PROCESS
import os

import mmcv
from mmcv.utils import deprecated_api_warning, is_tuple_of

def to_tensor(data,type='float'):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        if type=='float':
            return torch.from_numpy(data)
        elif type=='long':
            return torch.from_numpy(data).long()
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PROCESS.register_module
class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
        collect_keys (Sequence[str]): Keys that need to keep, but not to Tensor.
    """

    def __init__(self, keys=['img', 'mask'], collect_keys=[], long_keys=[],cfg=None):
        self.keys = keys
        self.collect_keys = collect_keys
        self.long_keys = long_keys

    def __call__(self, sample):
        data = {}
        if len(sample['img'].shape) < 3:
            sample['img'] = np.expand_dims(sample['img'], -1)
        for key in sample.keys():
            if key in self.keys:
                data[key] = to_tensor(sample[key])
            if key in self.collect_keys:
                data[key] = sample[key]
            if key in self.long_keys:
                data[key] = to_tensor(sample[key],type='long')
        data['img'] = data['img'].permute(2, 0, 1)
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'

@PROCESS.register_module
class DataToTensor(object):
    def __init__(self, keys,cfg=None):
        self.keys = keys

    def __call__(self, results):

        for key in self.keys:
            data = results[key]
            if key=='img':
                if len(data.shape) < 3:
                    data = np.expand_dims(data, -1)
                data = data.transpose(2, 0, 1)
            
            results[key] = to_tensor(data)
            
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'

@PROCESS.register_module
class ImageToTensor(object):
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys,cfg=None):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """

        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
            
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'

@PROCESS.register_module
class Collect(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "gt_semantic_seg".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the bottom/right
            if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg'),
                 cfg=None):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            if key not in results:
                img_meta[key] = None
            else:
                img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'

@PROCESS.register_module
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results,cfg=None):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)

        if 'event' in results:
            event = results['event']
            results['event'] = DC(to_tensor(event), stack=True, pad_dims=3)

        if 'seq_length' in results:
            seq_length = results['seq_length']
            results['seq_length'] = DC(to_tensor(seq_length), stack=True, pad_dims=None)

        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None,
                                                     ...].astype(np.int64)),
                stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__

''' 
    -------------------------------------------------------------------------------
    -------------------------------------------------------------------------------
    -------------------------------------------------------------------------------
'''

@PROCESS.register_module
class RandomLROffsetLABEL(object):
    def __init__(self,max_offset, cfg=None):
        self.max_offset = max_offset
    def __call__(self, sample):
        img = sample['img'] 
        label = sample['mask'] 
        offset = np.random.randint(-self.max_offset,self.max_offset)
        h, w = img.shape[:2]

        img = np.array(img)
        if offset > 0:
            img[:,offset:,:] = img[:,0:w-offset,:]
            img[:,:offset,:] = 0
        if offset < 0:
            real_offset = -offset
            img[:,0:w-real_offset,:] = img[:,real_offset:,:]
            img[:,w-real_offset:,:] = 0

        label = np.array(label)
        if offset > 0:
            label[:,offset:] = label[:,0:w-offset]
            label[:,:offset] = 0
        if offset < 0:
            offset = -offset
            label[:,0:w-offset] = label[:,offset:]
            label[:,w-offset:] = 0
        sample['img'] = img
        sample['mask'] = label
        
        return sample 

@PROCESS.register_module
class RandomUDoffsetLABEL(object):
    def __init__(self,max_offset, cfg=None):
        self.max_offset = max_offset
    def __call__(self, sample):
        img = sample['img'] 
        label = sample['mask'] 
        offset = np.random.randint(-self.max_offset,self.max_offset)
        h, w = img.shape[:2]

        img = np.array(img)
        if offset > 0:
            img[offset:,:,:] = img[0:h-offset,:,:]
            img[:offset,:,:] = 0
        if offset < 0:
            real_offset = -offset
            img[0:h-real_offset,:,:] = img[real_offset:,:,:]
            img[h-real_offset:,:,:] = 0

        label = np.array(label)
        if offset > 0:
            label[offset:,:] = label[0:h-offset,:]
            label[:offset,:] = 0
        if offset < 0:
            offset = -offset
            label[0:h-offset,:] = label[offset:,:]
            label[h-offset:,:] = 0
        sample['img'] = img
        sample['mask'] = label
        return sample 

@PROCESS.register_module
class Resize(object):
    def __init__(self, size, cfg=None):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, sample):
        out = list()
        sample['img'] = cv2.resize(sample['img'], self.size,
                              interpolation=cv2.INTER_CUBIC)
        if 'mask' in sample:
            sample['mask'] = cv2.resize(sample['mask'], self.size,
                                  interpolation=cv2.INTER_NEAREST)
        return sample


@PROCESS.register_module
class RandomCrop(object):
    def __init__(self, size, cfg=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        h, w = img_group[0].shape[0:2]
        th, tw = self.size

        out_images = list()
        h1 = random.randint(0, max(0, h - th))
        w1 = random.randint(0, max(0, w - tw))
        h2 = min(h1 + th, h)
        w2 = min(w1 + tw, w)

        for img in img_group:
            assert (img.shape[0] == h and img.shape[1] == w)
            out_images.append(img[h1:h2, w1:w2, ...])
        return out_images


@PROCESS.register_module
class CenterCrop(object):
    def __init__(self, size, cfg=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        h, w = img_group[0].shape[0:2]
        th, tw = self.size

        out_images = list()
        h1 = max(0, int((h - th) / 2))
        w1 = max(0, int((w - tw) / 2))
        h2 = min(h1 + th, h)
        w2 = min(w1 + tw, w)

        for img in img_group:
            assert (img.shape[0] == h and img.shape[1] == w)
            out_images.append(img[h1:h2, w1:w2, ...])
        return out_images

@PROCESS.register_module
class RandomRotation(object):
    def __init__(self, degree=(-10, 10), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST), padding=None, cfg=None):
        self.degree = degree
        self.interpolation = interpolation
        self.padding = padding
        if self.padding is None:
            self.padding = [0, 0]

    def _rotate_img(self, sample, map_matrix):
        h, w = sample['img'].shape[0:2]
        sample['img'] = cv2.warpAffine(
            sample['img'], map_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)

    def _rotate_mask(self, sample, map_matrix):
        if 'mask' not in sample:
            return
        h, w = sample['mask'].shape[0:2]
        sample['mask'] = cv2.warpAffine(
            sample['mask'], map_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)


    def __call__(self, sample):
        v = random.random()
        if v < 0.5:
            degree = random.uniform(self.degree[0], self.degree[1])
            h, w = sample['img'].shape[0:2]
            center = (w / 2, h / 2)
            map_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
            self._rotate_img(sample, map_matrix)
            self._rotate_mask(sample, map_matrix)
        return sample


@PROCESS.register_module
class RandomBlur(object):
    def __init__(self, applied, cfg=None):
        self.applied = applied

    def __call__(self, img_group):
        assert (len(self.applied) == len(img_group))
        v = random.random()
        if v < 0.5:
            out_images = []
            for img, a in zip(img_group, self.applied):
                if a:
                    img = cv2.GaussianBlur(
                        img, (5, 5), random.uniform(1e-6, 0.6))
                out_images.append(img)
                if len(img.shape) > len(out_images[-1].shape):
                    out_images[-1] = out_images[-1][...,
                                                    np.newaxis]  # single channel image
            return out_images
        else:
            return img_group


@PROCESS.register_module
class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy Image with a probability of 0.5
    """

    def __init__(self, cfg=None):
        pass

    def __call__(self, sample):
        v = random.random()
        if v < 0.5:
            sample['img'] = np.fliplr(sample['img'])
            if 'mask' in sample: sample['mask'] = np.fliplr(sample['mask'])
        return sample


@PROCESS.register_module
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """
    def __init__(self, img_norm, to_rgb=False, cfg=None):
        self.mean = np.array(img_norm['mean'], dtype=np.float32)
        self.std = np.array(img_norm['std'], dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, sample):
        # m = self.mean
        # s = self.std
        # img = sample['img'] 
        # if len(m) == 1:
        #     img = img - np.array(m)  # single channel image
        #     img = img / np.array(s)
        # else:
        #     img = img - np.array(m)[np.newaxis, np.newaxis, ...]
        #     img = img / np.array(s)[np.newaxis, np.newaxis, ...]
        # sample['img'] = img

        sample['img'] = mmcv.imnormalize(sample['img'], self.mean, self.std, self.to_rgb)
        sample['img_norm_cfg'] = dict(img_norm=dict(mean=self.mean, std=self.std), to_rgb=self.to_rgb)

        return sample 
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str


@PROCESS.register_module
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,cfg=None):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        # if random.randint(2):
        if random.randint(0, 1):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        # if random.randint(2):
        if random.randint(0, 1):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        # if random.randint(2):
        if random.randint(0, 1):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        # if random.randint(2):
        if random.randint(0, 1):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        # mode = random.randint(2)
        mode = random.randint(0, 1)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str