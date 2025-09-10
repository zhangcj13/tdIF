import os.path as osp
import numpy as np
import cv2
import os
import json
import torchvision
from .base_dataset import BaseDataset
from detection.utils.tusimple_metric import LaneEval
from .registry import DATASETS
import logging
import random
from tqdm import tqdm
from mmcv.parallel import DataContainer as DC

SPLIT_FILES = {
    'trainval': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}


@DATASETS.register_module
class TuSimple(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes, cfg)
        self.anno_files = SPLIT_FILES[split] 
        self.load_annotations()
        self.h_samples = list(range(160, 720, 10))

    def load_annotations(self):
        self.logger.info('Loading TuSimple annotations...')
        self.data_infos = []
        max_lanes = 0
        pbar = tqdm(total=10000)
        cnt = 0
        for anno_file in self.anno_files:   
            anno_file = osp.join(self.data_root, anno_file)
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))
                self.data_infos.append({
                    'img_path': osp.join(self.data_root, data['raw_file']),
                    'img_name': data['raw_file'],
                    'lanes': lanes,
                    'gt_lanes': gt_lanes,
                })
                pbar.update(1)
                pbar.set_description("Loading TuSimple annotations %d" % cnt)
                cnt+=1
        
        if self.training:
            random.shuffle(self.data_infos)
        self.max_lanes = max_lanes

    def pred2lanes(self, pred):
        ys = np.array(self.h_samples) / self.cfg.ori_img_h
        lanes = []
        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * self.cfg.ori_img_w).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())

        return lanes
    
    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        if not osp.isfile(data_info['img_path']):
            raise FileNotFoundError('cannot find file: {}'.format(data_info['img_path']))

        img = cv2.imread(data_info['img_path'])

        img = img[self.cfg.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({'img': img})

        sample = self.processes(sample)
        meta = {'full_img_path': data_info['img_path'],
                'img_name': data_info['img_name']}
        meta = DC(meta, cpu_only=True)
        sample.update({'meta': meta})
        
        return sample 

    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self.data_infos[idx]['img_name']
        lanes = self.pred2lanes(pred)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions, runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def evaluate(self, predictions, output_basedir, runtimes=None, return_metric=False):
        table=None
        if type(predictions) == dict:
            predictions = predictions['lane_line']
            table={}

        pred_filename = os.path.join(output_basedir, 'tusimple_predictions.json')
        self.save_tusimple_predictions(predictions, pred_filename, runtimes)
        # result, acc = LaneEval.bench_one_submit(pred_filename, self.cfg.test_json_file)
        result, acc = LaneEval.bench_one_submit2gt(pred_filename, self.data_infos, y_samples=self.h_samples)
        if return_metric:
            return result
        self.logger.info(result)
        if table is None:
            return acc
        else:
            table['lane_acc'] = acc
            return table
