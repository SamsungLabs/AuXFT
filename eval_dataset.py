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

import warnings
import argparse
import torch

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

from models import YoloFeats
from utils import get_train_val_loaders, Metrics, clean_predictions

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=ZeroDivisionError)

def run_inference(args, loader, model):
    """
        compute metrics
    """
    acc = 0
    cts = 0
    with torch.inference_mode():
        metrics = Metrics(loader.dataset.names, conf=0.001)
        for sample in tqdm(loader, ncols=120):
            x = sample['img'] / 255.

            if args.debug:
                _, ax = plt.subplots(1,1)
                ax.imshow(x[0].cpu().permute(1,2,0))

                gh, gw = x.shape[2:]
                cx, cy, w, h = sample['bboxes'][0]
                x0, y0 = gw*(cx-w/2).item(), gh*(cy-h/2).item()
                x1, y1 = gw*(cx+w/2).item(), gh*(cy+h/2).item()
                ax.add_patch(Rectangle((x0, y0), w.item()*gw, h.item()*gh, fill=False, color='g'))
                ax.text(x1, y0, loader.dataset.names[str(sample['cls'][0].int().item())],
                        verticalalignment='top', horizontalalignment='right',
                        bbox={'facecolor': 'g', 'edgecolor': 'g', 'pad': 0})

            x = x.to(args.device, dtype=torch.float32)

            (pred, _), _ = model(x)

            boxes = model.get_results(pred)
            for i, box in enumerate(boxes):
                if args.debug:
                    if i == 0:
                        for x0, y0, x1, y1, conf, cls in box.cpu():
                            if conf > .01:
                                ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                                                       fill=False, color='r'))
                                ax.text(x0, y0, loader.dataset.data\
                                        ['coarse_names'][str(cls.int().item())],
                                        verticalalignment='top', horizontalalignment='left',
                                        bbox={'facecolor': 'r', 'edgecolor': 'r', 'pad': 0})
                box, labels, cls = clean_predictions(box, sample, i)
                metrics(box, labels, cls)

                acc += any(torch.any(box[:,-1].cpu() == cl) for cl in cls)
                cts += 1

            if args.debug:
                plt.show()

    map50, _, map50_95 = metrics.get_ap()
    if args.use_map50:
        return map50, 100*acc/cts
    return map50_95, 100*acc/cts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model-related arguments
    parser.add_argument('--ckpt', default='ckpts/auxft.pth')

    # dataset-related arguments
    parser.add_argument('--dataset', default='perseg',
                        choices=['mixed', 'pod', 'perseg', 'core50', 'icub'])
    parser.add_argument('--val_mode', default=3, type=int)
    parser.add_argument('--use_map50', action='store_true')

    # training arguments
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')

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
    g_args.episodic = False

    tloader, vloader = get_train_val_loaders(g_args)

    g_model = YoloFeats(nc=31,
                        is_base=True)
    g_model.load_state_dict({k.replace('module.', ''):v for k,v
                            in torch.load(g_args.ckpt, map_location='cpu').items()}, strict=False)
    g_model.to(g_args.device)
    g_model.eval()

    # inference on training set
    print("Start Evaluation on Training Set")
    tamp, tacc = run_inference(g_args, tloader, g_model)
    print("mAP50-95: %.2f, Accuracy: %.2f"%(tamp, tacc))

    if vloader is not None:
        # inference on validation set
        print("Start Evaluation on Validation Set")
        vamp, vacc = run_inference(g_args, vloader, g_model)
        print("mAP50-95: %.2f, Accuracy: %.2f"%(vamp, vacc))
