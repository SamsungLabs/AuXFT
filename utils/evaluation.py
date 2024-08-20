################################################################################
# Part of the code adapted from: https://github.com/ultralytics/ultralytics)
# Copyright (c) 2023.
#
# Copyright (c) 2024 Samsung Electronics Co., Ltd.
#
# Author(s):
# Francesco Barbato (f.barbato@samsung.com; francesco.barbato@dei.unipd.it)
# Umberto Michieli (u.michieli@samsung.com)
# Jijoong Moon (jijoong.moon@samsung.com)
# Pietro Zanuttigh (zanuttigh@dei.unipd.it)
# Mete Ozay (m.ozay@samsung.com)
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# For conditions of distribution and use, see the accompanying LICENSE.md file.
################################################################################

import numpy as np
import torch

from ultralytics.utils.metrics import ConfusionMatrix, box_iou, ap_per_class
from ultralytics.utils.ops import scale_boxes, xywh2xyxy

def clean_predictions(box, sample, i):
    """
        rescale bounding boxes and remove illegal classes
    """
    idx = sample['batch_idx'] == i

    h0, w0 = sample['ori_shape'][i]
    h1, w1 = sample['img'][i].shape[1:]

    cls = sample['cls'][idx]

    box[:,:4] = scale_boxes(
                        (h1,w1),
                        box[:,:4],
                        (h0,w0),
                        sample['ratio_pad'][i]
                    )
    labels = scale_boxes(
                        (h1,w1),
                        xywh2xyxy(sample['bboxes'][idx])*torch.tensor((w1,h1,w1,h1)),
                        (h0,w0),
                        sample['ratio_pad'][i]
                    )
    labels = torch.cat((cls, labels), 1)
    return box[box[:, -1]>=0], labels.to(box.device), cls.squeeze(-1)

class Metrics():
    """
        metrics class, from ultralytics
    """
    def __init__(self, names, conf):
        self.nc = len(names)
        self.names = names
        self.conf = conf

        self.cm = ConfusionMatrix(nc=self.nc, conf=self.conf)

        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

        self.stats = []

    def __call__(self, box, labels, cls):
        # aggregate metrics
        self.cm.process_batch(box, labels)

        # compute per-batch statistics
        iou = box_iou(labels[:, 1:], box[:, :4])
        correct_bboxes = self.match_predictions(box[:, 5], labels[:, 0], iou)
        self.stats.append((correct_bboxes, box[:, 4], box[:, 5], cls))

    def get_ap(self):
        """
            parse stats to get maps
        """
        stats = self.get_all_stats()
        if stats is None:
            return 0, 0, 0
        ap = 100*stats[5]
        map50 = ap[:,0].mean()
        map75 = ap[:,5].mean()
        map50_95 = ap.mean()
        return map50, map75, map50_95

    def get_map(self):
        """
            get global map
        """
        tp, fp = self.cm.tp_fp()
        return 100*np.mean(tp/(tp+fp+1e-5))

    def get_iap(self):
        """
            get instance map
        """
        stats = self.get_all_stats()
        ap = 100*stats[5].mean(axis=1)
        return ap

    def print_ap(self, ap=None, std=None):
        """
            print summary
        """
        stats = self.get_all_stats()
        if ap is None:
            ap = 100*stats[5].mean(axis=1)
        names = [self.names[str(cid)] for cid in stats[6]]
        if std is None:
            for name, p in zip(names, ap):
                print('% 20s \t %06.3f'%(name, p))
        else:
            for name, p, s in zip(names, ap, std):
                print('% 20s \t %06.3f +/- %06.3f'%(name, p, s))

    def get_all_stats(self):
        """
        Returns:
        (tuple): A tuple of six arrays and one array of unique classes, where:
            tp (np.ndarray): True positive counts at threshold 
                given by max F1 metric for each class.Shape: (nc,).
            fp (np.ndarray): False positive counts at threshold given by max 
                F1 metric for each class. Shape: (nc,).
            p (np.ndarray): Precision values at threshold given by max F1 metric 
                for each class. Shape: (nc,).
            r (np.ndarray): Recall values at threshold given by max F1 metric 
                for each class. Shape: (nc,).
            f1 (np.ndarray): F1-score values at threshold given by max F1 metric 
                for each class. Shape: (nc,).
            ap (np.ndarray): Average precision for each class at different IoU 
                thresholds. Shape: (nc, 10).
            unique_classes (np.ndarray): An array of unique classes that have data. Shape: (nc,).
            p_curve (np.ndarray): Precision curves for each class. Shape: (nc, 1000).
            r_curve (np.ndarray): Recall curves for each class. Shape: (nc, 1000).
            f1_curve (np.ndarray): F1-score curves for each class. Shape: (nc, 1000).
            x (np.ndarray): X-axis values for the curves. Shape: (1000,).
            prec_values: Precision values at mAP@0.5 for each class. Shape: (nc, 1000).
        """
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        if len(stats) and stats[0].any():
            return ap_per_class(*stats, names=self.names)
        return None

    def match_predictions(self, pred_classes, true_classes, iou):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values 
                for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)
