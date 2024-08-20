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


from copy import deepcopy
from typing import Any, Mapping

from scipy.stats import skew

import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.nn import DetectionModel
from ultralytics.nn.modules import Detect, Segment, Pose
from ultralytics.nn.tasks import yaml_model_load, parse_model
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, IterableSimpleNamespace
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import initialize_weights

class YoloFeats(DetectionModel):
    """
        yolo with features
    """
    def __init__(self,
                 cfg='yolov8n.yaml',
                 ch=3,
                 nc=None,
                 verbose=False,
                 use_fcn=False,
                 dinoc=384,
                 dinos=48,
                 is_base=False):  # model, input channels, number of classes

        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super(DetectionModel, self).__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        self.overrides = {}  # overrides for trainer objec
        self.overrides['model'] = cfg

        # Below added to allow export from YAMLs
        self.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, **self.overrides})  # combine default and model args (prefer model args)

        self.dinoc = dinoc
        self.dinos = dinos
        self.nc = nc
        self.use_fcn = use_fcn
        self.is_base = is_base

        if not self.is_base:
            if self.use_fcn:
                self.fmap11 = nn.Conv2d( 64, 4*64, 3, padding=1, bias=False)
                self.fmap12 = nn.Conv2d(128, 4*128, 3, padding=1, bias=False)
                self.fmap13 = nn.Conv2d(256, 4*256, 3, padding=1, bias=False)

                self.fmap21 = nn.Conv2d(4*64, dinoc, 3, padding=1, bias=False)
                self.fmap22 = nn.Conv2d(4*128, dinoc, 3, padding=1, bias=False)
                self.fmap23 = nn.Conv2d(4*256, dinoc, 3, padding=1, bias=False)

                self.relu = nn.ReLU()
            else:
                self.fmap1 = nn.Conv2d( 64, dinoc, 3, padding=1, bias=False)
                self.fmap2 = nn.Conv2d(128, dinoc, 3, padding=1, bias=False)
                self.fmap3 = nn.Conv2d(256, dinoc, 3, padding=1, bias=False)

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info("Overriding model.yaml nc=%s with nc=%d", self.yaml['nc'], nc) # MODIFIED
            self.yaml['nc'] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0]
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def get_results(self, preds, conf=0.001, iou_thres=0.6):
        """
            non max suppression wrapper
        """
        return non_max_suppression(preds,
                                   conf,
                                   iou_thres,
                                   agnostic=self.args.agnostic_nms,
                                   max_det=self.args.max_det,
                                   classes=self.args.classes)

    def _predict_once(self, x, profile=False, visualize=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt = [], []  # outputs
        fs = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if 'Detect' in m.__class__.__name__:
                fs = [xi for xi in x]
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        if self.is_base:
            return x, fs

        yf1, yf2, yf3 = fs
        if self.use_fcn:
            yf1 = self.fmap11(yf1)
            yf1 = self.fmap21(self.relu(yf1))

            yf2 = self.fmap12(yf2)
            yf2 = self.fmap22(self.relu(yf2))

            yf3 = self.fmap13(yf3)
            yf3 = self.fmap23(self.relu(yf3))
        else:
            yf1 = self.fmap1(yf1)
            yf2 = self.fmap2(yf2)
            yf3 = self.fmap3(yf3)

        yf1 = F.interpolate(yf1, (self.dinos, self.dinos), mode='area')                         # downsample
        yf2 = F.interpolate(yf2, (self.dinos, self.dinos), mode='bilinear', align_corners=True) # ~ resample (42x42 vs. 48x48)
        yf3 = F.interpolate(yf3, (self.dinos, self.dinos), mode='bicubic', align_corners=True)  # upsample

        return x, (yf1, yf2, yf3)

class ColMap(nn.Module):
    """
        feats->rgb->feats
    """
    def __init__(self):
        super().__init__()

        self.fmap = nn.Conv2d(384, 3, 1, bias=False)
        self.ifmap = nn.Conv2d(3, 384, 1, bias=False)
        self.sigma = nn.Sigmoid()

    def forward(self, x):
        """
            torch module forward
        """
        rgb = self.sigma(self.fmap(x))
        rx = self.ifmap(rgb)
        return rgb, rx

    def color_features(self, feat):
        """
            color features
        """
        with torch.no_grad():
            return self.sigma(self.fmap(feat))

class DinoFeats(nn.Module):
    """
        dino features module
    """
    def __init__(self,
                 yoloc=(64,128,256),
                 yolos=(28, 14, 7),
                 mode='conv'):
        super().__init__()

        self.yoloc = yoloc
        self.yolos = yolos
        self.mode = mode

        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False

        self.colmap = ColMap()

    def train(self, *args, **kwargs):
        o = super().train(*args, **kwargs)
        self.dino.eval()
        return o

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        o = super().load_state_dict(state_dict, strict, assign)
        # reset dino for safety
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False
        return o

    def forward(self, x):
        """
            torch module forward
        """
        B, _, H, W = x.shape
        H1, W1 = H//14, W//14
        with torch.no_grad():
            o = self.dino(x, is_training=True)["x_prenorm"]
            #x, c = o["x_prenorm"], o['x_norm_clstoken']
            x, c = o[:, self.dino.num_register_tokens + 1 :], o[:,self.dino.num_register_tokens]
            x = x.permute(0,2,1).reshape(B,384,H1,W1)

        _, rx = self.colmap(x)
        return x, rx, c

    def color_features(self, feat):
        """
            color features
        """
        return self.colmap.color_features(feat)

class YoloVecs(YoloFeats):
    """
        yolo with vectors in output
    """
    def __init__(self,
                 cfg='yolov8n.yaml',
                 ch=3,
                 nc=None,
                 verbose=False,
                 use_gt_boxes=False,
                 use_fcn=False,
                 dinoc=384,
                 dinos=48,
                 is_base=False,
                 concatenate_chs=True,
                 mask_extra_coarse=False,
                 dataset=None,
                 pool_mode='mean'):
        self.mask_extra_coarse = mask_extra_coarse

        if mask_extra_coarse:
            assert dataset is not None, "A dataset object is needed to mask the extra coarse classes"
            self.ids_to_mask = [i+4 for i in range(nc) if i not in dataset.valid_coarse] # shift indices to class channels

        super().__init__(cfg, ch, nc, verbose, use_fcn, dinoc, dinos, is_base)
        self.use_gt_boxes = use_gt_boxes
        self.concatenate_chs = concatenate_chs
        self.pool_mode = pool_mode

    def __call__(self, x, conf=.3, sample=None) -> Any:
        assert not self.training, "To use this model only works in evaluation mode"
        (pred, _), feats =  super().__call__(x)

        if self.mask_extra_coarse:
            pred[:,self.ids_to_mask] = 0

        if not self.is_base:
            feat = feats[0] + feats[1] + feats[2]
        if self.use_gt_boxes and sample is not None:
            boxlist = []
            for i in range(x.shape[0]):
                H1, W1 = sample['img'][i].shape[1:]
                idx = sample['batch_idx'] == i
                box = xywh2xyxy(sample['bboxes'][idx])*torch.tensor((W1,H1,W1,H1))
                boxlist.append(torch.cat([box, torch.ones_like(sample['cls'][idx]), sample['cls'][idx]], dim=1).to(device=x.device))

        else:
            boxlist = self.get_results(pred, conf=conf)

        return self.cut_features_base(boxlist, feats, x.shape[2:]) if self.is_base \
                    else self.cut_features(boxlist, feat, x.shape[2:]), boxlist

    def cut_features(self, boxlist, feat, oshape):
        """
            cut features and collapse them
        """
        batch = []
        for i, boxes in enumerate(boxlist):
            obox = []
            for box in boxes:
                conf, clas = box[4].item(), int(box[5])

                fshape = feat.shape[2:]
                scale = torch.tensor([fshape[0]/oshape[0], fshape[1]/oshape[1], fshape[0]/oshape[0], fshape[1]/oshape[1]], device=feat.device)
                coords = box[:4]*scale
                coords[[0,1]], coords[[2,3]] = torch.floor(coords[[0,1]]), torch.ceil(coords[[2,3]])
                coords = torch.clamp_min(coords.int(), 0)

                cut = feat[i,:,coords[1]:coords[3]+1, coords[0]:coords[2]+1].reshape(feat.shape[1],-1)
                if self.pool_mode == 'mean':
                    obox.append([cut.mean(dim=-1), conf, clas])
                elif self.pool_mode == 'max':
                    obox.append([cut.max(dim=-1)[0], conf, clas])
                elif self.pool_mode == 'median':
                    obox.append([cut.median(dim=-1)[0], conf, clas])
                elif self.pool_mode == 'std':
                    obox.append([torch.cat([cut.mean(dim=-1), cut.std(dim=-1)]), conf, clas])
                elif self.pool_mode == 'skew':
                    obox.append([torch.cat([cut.mean(dim=-1), cut.std(dim=-1), torch.from_numpy(skew(cut.cpu().numpy(), axis=-1)).to(device=cut.device)]), conf, clas])
                else:
                    raise ValueError('Unknown pooling strategy {%s}'%self.pool_mode)

            batch.append(obox)
        return batch

    def cut_features_base(self, boxlist, feats, oshape):
        """
            cut features with concat
        """
        batch = []
        for i, boxes in enumerate(boxlist):
            obox = []
            for box in boxes:
                conf, clas = box[4].item(), int(box[5])
                if self.concatenate_chs:
                    ofeat = []
                    for feat in feats:
                        fshape = feat.shape[2:]
                        scale = torch.tensor([fshape[0]/oshape[0], fshape[1]/oshape[1], fshape[0]/oshape[0], fshape[1]/oshape[1]], device=feat.device)
                        coords = box[:4]*scale
                        coords[[0,1]], coords[[2,3]] = torch.floor(coords[[0,1]]), torch.ceil(coords[[2,3]])
                        coords = torch.clamp_min(coords.int(), 0)

                        cut = feat[i,:,coords[1]:coords[3]+1, coords[0]:coords[2]+1].reshape(feat.shape[1],-1)
                        if self.pool_mode == 'mean':
                            ofeat.append(cut.mean(dim=-1))
                        elif self.pool_mode == 'max':
                            ofeat.append(cut.max(dim=-1)[0])
                        elif self.pool_mode == 'median':
                            ofeat.append(cut.median(dim=-1)[0])
                        elif self.pool_mode == 'std':
                            ofeat.append(torch.cat([cut.mean(dim=-1), cut.std(dim=-1)]))
                        elif self.pool_mode == 'skew':
                            ofeat.append(torch.cat([cut.mean(dim=-1), cut.std(dim=-1), torch.from_numpy(skew(cut.cpu().numpy(), axis=-1)).to(device=cut.device)]))
                        else:
                            raise ValueError('Unknown pooling strategy {%s}'%self.pool_mode)

                    obox.append([torch.cat(ofeat, dim=0), conf, clas])
                else:
                    feat = feats[2]
                    fshape = feat.shape[2:]
                    scale = torch.tensor([fshape[0]/oshape[0], fshape[1]/oshape[1], fshape[0]/oshape[0], fshape[1]/oshape[1]], device=feat.device)
                    coords = box[:4]*scale
                    coords[[0,1]], coords[[2,3]] = torch.floor(coords[[0,1]]), torch.ceil(coords[[2,3]])
                    coords = torch.clamp_min(coords.int(), 0)

                    cut = feat[i,:,coords[1]:coords[3]+1, coords[0]:coords[2]+1].reshape(feat.shape[1],-1)
                    if self.pool_mode == 'mean':
                        obox.append([cut.mean(dim=-1), conf, clas])
                    elif self.pool_mode == 'max':
                        obox.append([cut.max(dim=-1)[0], conf, clas])
                    elif self.pool_mode == 'median':
                        obox.append([cut.median(dim=-1)[0], conf, clas])
                    elif self.pool_mode == 'std':
                        obox.append([torch.cat([cut.mean(dim=-1), cut.std(dim=-1)]), conf, clas])
                    elif self.pool_mode == 'skew':
                        obox.append([torch.cat([cut.mean(dim=-1), cut.std(dim=-1), torch.from_numpy(skew(cut.cpu().numpy(), axis=-1)).to(device=cut.device)]), conf, clas])
                    else:
                        raise ValueError('Unknown pooling strategy {%s}'%self.pool_mode)
            batch.append(obox)
        return batch

class DinoVecs(nn.Module):
    """
        dino (oracle single) embedding vectors
    """
    def __init__(self,
                 yolo,
                 use_gt_boxes=False):
        super().__init__()

        self.use_gt_boxes = use_gt_boxes

        self.yolo = yolo
        self.yolo.eval()

        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dino.eval()

    def forward(self, x, conf=.3, sample=None):
        """
            pytorch forward pass
        """
        B, _, H, W = x.shape
        H1, W1 = H//14, W//14

        (pred, _), _ = self.yolo(x)
        feats = self.dino(x, is_training=True)["x_norm_patchtokens"]
        feats = feats.permute(0,2,1).reshape(B,384,H1,W1)

        if self.use_gt_boxes and sample is not None:
            boxlist = []
            for i in range(x.shape[0]):
                H1, W1 = sample['img'][i].shape[1:]
                idx = sample['batch_idx'] == i
                box = xywh2xyxy(sample['bboxes'][idx])*torch.tensor((W1,H1,W1,H1))
                boxlist.append(torch.cat([box, torch.ones_like(sample['cls'][idx]), sample['cls'][idx]], dim=1).to(device=x.device))
        else:
            boxlist = self.yolo.get_results(pred, conf=conf)

        return self.cut_features(boxlist, feats, x.shape[2:]), boxlist

    def cut_features(self, boxlist, feat, oshape):
        """
            cut features
        """
        batch = []
        for i, boxes in enumerate(boxlist):
            obox = []
            for box in boxes:
                conf, clas = box[4].item(), int(box[5])
                fshape = feat.shape[2:]
                scale = torch.tensor([fshape[0]/oshape[0], fshape[1]/oshape[1], fshape[0]/oshape[0], fshape[1]/oshape[1]], device=feat.device)
                coords = box[:4]*scale
                coords[[0,1]], coords[[2,3]] = torch.floor(coords[[0,1]]), torch.ceil(coords[[2,3]])
                coords = torch.clamp_min(coords.int(), 0)
                ofeat = feat[i,:,coords[1]:coords[3]+1, coords[0]:coords[2]+1].mean([1,2])
                obox.append([ofeat, conf, clas])
            batch.append(obox)
        return batch

class DemoVecs(nn.Module):
    """
        oracle multi
    """
    def __init__(self, yolo, use_gt_boxes=False):
        super().__init__()

        self.use_gt_boxes = use_gt_boxes

        self.yolo = yolo
        self.yolo.eval()

        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dino.eval()

    def forward(self, x, conf=.3, sample=None):
        """
            forward pass
        """
        _, _, H, W = x.shape
        H1, W1 = H//14, W//14

        (pred, _), _ = self.yolo(x)

        if self.use_gt_boxes and sample is not None:
            boxlist = []
            for i in range(x.shape[0]):
                H1, W1 = sample['img'][i].shape[1:]
                idx = sample['batch_idx'] == i
                box = xywh2xyxy(sample['bboxes'][idx])*torch.tensor((W1,H1,W1,H1))
                boxlist.append(torch.cat([box, torch.ones_like(sample['cls'][idx]), sample['cls'][idx]], dim=1).to(device=x.device))
        else:
            boxlist = self.yolo.get_results(pred, conf=conf)

        return self.cut_images(boxlist, x)

    def cut_images(self, boxlist, x):
        """
            cut image slices from bboxes
        """
        batch = []
        for i, boxes in enumerate(boxlist):
            obox = []
            for box in boxes:
                conf, clas = box[4].item(), int(box[5])
                coords = box[:4]
                coords[[0,1]], coords[[2,3]] = torch.floor(coords[[0,1]]), torch.ceil(coords[[2,3]])
                coords = torch.clamp_min(coords.int(), 0)
                ix = F.interpolate(x[i:i+1,:,coords[1]:coords[3]+1, coords[0]:coords[2]+1], (224,224))
                ofeat = self.dino(ix, is_training=True)["x_norm_patchtokens"][0].mean(dim=0)
                obox.append([ofeat, conf, clas])
            batch.append(obox)
        return batch, boxlist
