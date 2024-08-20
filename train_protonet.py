################################################################################
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

import argparse
import warnings
import time
from copy import deepcopy

import torch
import numpy as np
from tqdm import trange

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from utils import Metrics, clean_predictions, get_model_and_protonet, get_train_val_loaders

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=ZeroDivisionError)

def str2bool(s):
    """
        string to bool
    """
    s = s.lower()
    if s in ['1', 't', 'true']:
        return True
    if s in ['0', 'f', 'false']:
        return False
    raise ValueError(f"[{s}] cannot be parsed as boolean")

def train_protonet(tloader, args, model, proto):
    """
        train protonet
    """
    for sample in tloader:
        x = sample['img'] / 255.
        x = x.to(args.device, dtype=torch.float32)

        if args.square_boxes:
            for bid, box in enumerate(sample['bboxes']):
                cx, cy, w, h = box
                l = args.expand_rate*(w+h)/2
                sample['bboxes'][bid] = torch.tensor([cx, cy, l, l])

        if model.use_gt_boxes:
            ssample = deepcopy(sample)
            for i in range(ssample['cls'].shape[0]):
                ssample['cls'][i] = int(
                    tloader.dataset.idmap[str(ssample['cls'][i].int().item())])
        else:
            ssample = None

        vecs, _ = model(x, conf=.3, sample=ssample)
        proto.train_protos(vecs, sample['cls'].int())

def plot_image_and_gt_box(x, sample, vloader):
    """
        debug plots - image and gt box
    """
    fig, ax = plt.subplots(1,1)
    ax.imshow(x[0].cpu().permute(1,2,0))

    gh, gw = x.shape[2:]
    cx, cy, w, h = sample['bboxes'][0]
    x0, y0 = gw*(cx-w/2).item(), gh*(cy-h/2).item()
    x1 = gw*(cx+w/2).item()
    ax.add_patch(Rectangle((x0, y0), w.item()*gw, h.item()*gh, fill=False, color='g'))
    ax.text(x1, y0, vloader.dataset.names[str(sample['cls'][0].int().item())],
            verticalalignment='top', horizontalalignment='right',
            bbox={'facecolor': 'g', 'edgecolor': 'g', 'pad': 0})
    return fig, ax

def plot_predictions(box, ax, vloader):
    """
        debug plots - predicted boxes
    """
    for x0, y0, x1, y1, conf, cls in box.cpu():
        if conf > .01:
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, color='r'))
            if cls.int().item() > 0:
                ax.text(x0, y0, vloader.dataset.names[str(cls.int().item())],
                        verticalalignment='top', horizontalalignment='left',
                        bbox={'facecolor': 'r', 'edgecolor': 'r', 'pad': 0})
            else:
                ax.text(x0, y0, 'None',
                        verticalalignment='top', horizontalalignment='left',
                        bbox={'facecolor': 'r', 'edgecolor': 'r', 'pad': 0})

def eval_protonet(vloader, args, model, proto, metrics):
    """
        evaluate protonet
    """
    acc = 0
    cts = 0
    for sid, sample in enumerate(vloader):
        x = sample['img'] / 255.
        x = x.to(args.device, dtype=torch.float32)

        if args.square_boxes:
            for bid, box in enumerate(sample['bboxes']):
                cx, cy, w, h = box
                l = args.expand_rate*(w+h)/2
                sample['bboxes'][bid] = torch.tensor([cx, cy, l, l])

        vecs, preds = model(x, conf=0.001)
        preds = proto(vecs, preds)

        if args.debug or args.save_images:
            fig, ax = plot_image_and_gt_box(x, sample, vloader)

        for i, box in enumerate(preds):
            if args.debug or args.save_images:
                plot_predictions(box, ax, vloader)

            # this also removes boxes that didn't get their label changed
            box, labels, cls = clean_predictions(box, sample, i)
            metrics(box, labels, cls)

            acc += any(torch.any(box[:,-1].cpu() == cl) for cl in cls)
            cts += 1

        if args.debug:
            plt.show()
        if args.save_images:
            fig.tight_layout()
            fig.savefig('images_dump/%04d.png'%sid)
            plt.close()
    return acc, cts

def run_episode(args, tloader, vloader, model, proto, verbose=False):
    """
        run one episodic training and get the metrics
    """
    max_vram = 0

    if args.save_images:
        matplotlib.use('webagg')

    with torch.inference_mode():
        metrics = Metrics(vloader.dataset.names, conf=0.001)
        strain = time.time()
        train_protonet(tloader, args, model, proto)
        max_vram += torch.cuda.max_memory_reserved(args.device_id)/1024/1024
        etrain = time.time()
        acc, cts = eval_protonet(vloader, args, model, proto, metrics)
        evalid = time.time()
        max_vram += torch.cuda.max_memory_reserved(args.device_id)/1024/1024

        map50, _, map50_95 = metrics.get_ap()
        if verbose:
            metrics.print_ap()

        if args.use_map50:
            return map50, 100*acc/cts, (etrain-strain)/len(tloader.dataset), \
                (evalid-etrain)/len(vloader.dataset), max_vram/2
        return map50_95, 100*acc/cts, (etrain-strain)/len(tloader.dataset), \
            (evalid-etrain)/len(vloader.dataset), max_vram/2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model-related arguments
    parser.add_argument('--model', default='residual',
                        choices=['base', 'residual', 'dino', 'demo'],
                        help='Which model configuration to use')
    parser.add_argument('--pnet', default='cond',
                        choices=['cond', 'base', 'simple'],
                        help="Which protonet configuration to use")
    parser.add_argument('--pool_mode', default='mean',
                        choices=['mean', 'median', 'std', 'skew', 'max'],
                        help="Which pooling strategy to use")
    parser.add_argument('--use_fcn', action='store_true',
                        help="Whether to use the 'fcn' configuration for AuXFT")
    parser.add_argument('--mask_extra', action='store_true',
                        help="Whether to mask extra classes when computing protonet distribution")
    parser.add_argument('--cat_chs', action='store_true',
                        help="Whether to use the 'cat' configuration for the baseline")
    parser.add_argument('--coarse_disabled', type=str2bool, default=True,
                        help="Whether to consider or not the coarse classes in the protonet output")
    parser.add_argument('--ckpt', default='ckpts/auxft.pth',
                        help="The checkpoint to be loaded, must match the configuration provided in --model")
    parser.add_argument('--base_ckpt', default='ckpts/base.pth',
                        help="The checkpoint used when --model=base")
    parser.add_argument('--use_map50', action='store_true',
                        help="Whether to measure mAP50 instead of mAP50-95")

    # dataset-related arguments
    parser.add_argument('--dataset', default='perseg',
                        choices=['mixed', 'pod', 'perseg', 'core50', 'icub'],
                        help="Which dataset to use for evaluation")
    parser.add_argument('--support', default=1, type=int,
                        help="Size of the support set for each episode")
    parser.add_argument('--val_mode', default=3, type=int,
                        help="Which validation set to use, only relevant for POD")
    parser.add_argument('--episodic', type=str2bool, default=True,
                        help="Whether to run the evaluation in episodic mode")
    parser.add_argument('--episodes', default=100, type=int,
                        help="Number of episodes")


    # training arguments
    parser.add_argument('--device', default='cuda', help="Pytorch device")
    parser.add_argument('--device_id', default=0, type=int, help="Pytorch device id, relevant for multi-GPU machines")
    parser.add_argument('--verbose', action='store_true', help="Print per-class AP")
    parser.add_argument('--debug', action='store_true', help="Show predictions on matplotlib")
    parser.add_argument('--save_images', action='store_true', help="Save predictions as images")
    parser.add_argument('--square_boxes', action='store_true', help="Convert GT boxes to square, see Fig. 4 of the paper")
    parser.add_argument('--expand_rate', type=float, default=1, help="Expansion rate for the GT boxes, see Fig. 4 of the paper")

    g_args = parser.parse_args()

    # set cuda device
    if g_args.device == 'cuda':
        g_args.device += ':%d'%g_args.device_id

    print("*"*100)
    print("*"+" "*29+"Running with the following configuration:"+" "*28+"*")
    print("* % 30s: % 64s *"%('Argument', 'Value'))
    print("*"*100)
    for k, v in vars(g_args).items():
        print("* % 30s: % 64s *"%(k,v))
    print("*"*100, '\n\n')

    g_args.coarse_labels = True

    g_tloader, g_vloader = get_train_val_loaders(g_args)
    g_model, g_proto = get_model_and_protonet(g_args, g_tloader.dataset)

    gmap = []
    gacc = []
    gttime = []
    gvtime = []
    gvram = []

    pbar = trange(g_args.episodes if g_args.episodic and not g_args.debug \
                and not g_args.save_images else 1,
                desc='Avg. mAP: %05.2f, Avg. Acc: %05.2f, Episode mAP: %05.2f, Episode Acc: %05.2f, Training Time: %06.4fs/im, Validation Time: %06.4fs/im, Max VRAM: %.2fMB'%(0,0,0,0,0,0,0), leave=False, ncols=200)
    for ep in pbar:
        if g_args.episodic:
            g_tloader.dataset.init_episode(ep)
            g_vloader.dataset.init_episode(ep)
            g_proto.reset()

        emAP, eacc, ettime, evtime, evram = run_episode(
            g_args, g_tloader, g_vloader, g_model, g_proto,
            verbose=g_args.verbose and not g_args.episodic)
        
        gmap.append(emAP)
        gacc.append(eacc)
        gttime.append(ettime)
        gvtime.append(evtime)
        gvram.append(evram)

        pbar.set_description('Avg. mAP: %05.2f, Avg. Acc: %05.2f, Episode mAP: %05.2f, Episode Acc: %05.2f, Training Time: %06.4fs/im, Validation Time: %06.4fs/im, Max VRAM: %.2fMB'%(np.mean(gmap), np.mean(gacc), emAP, eacc, ettime, evtime, evram))

    print("-"*100)
    print("Average mAP50-95: %.2f, Standard Deviation: %.2f"%(np.mean(gmap), np.std(gmap)))
    print("Average Accuracy: %.2f, Standard Deviation: %.2f"%(np.mean(gacc), np.std(gacc)))
    print("Average Training time: %.4fs/im, Standard Deviation: %.4f"%(
                                                          np.mean(gttime), np.std(gttime)))
    print("Average Inference time: %.4fs/im, Standard Deviation: %.4f"%(
                                                          np.mean(gvtime), np.std(gvtime)))
    print("Average Max Reserved VRAM: %fMB, Standard Deviation: %.2f"%(
                                                          np.mean(gvram), np.std(gvram)))
    print("-"*100)
