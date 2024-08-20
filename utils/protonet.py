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

import torch

from models import YoloFeats, YoloVecs, DinoVecs, DemoVecs, Conditional, BaseProtonet, SimpleShot

def get_model_and_protonet(args, dataset):
    """
        parse arguments to get protonet and detector
    """
    if args.pnet == 'cond':
        pnet = Conditional
    elif args.pnet == 'base':
        pnet = BaseProtonet
    elif args.pnet == 'simple':
        pnet = SimpleShot

    if args.model == 'base':
        yolo = YoloVecs(nc=31,
                        use_gt_boxes=True,
                        use_fcn=args.use_fcn,
                        is_base=True,
                        concatenate_chs=args.cat_chs,
                        mask_extra_coarse=args.mask_extra,
                        dataset=dataset,
                        pool_mode=args.pool_mode)
        odict = dict(yolo.state_dict()) # dict() needed to silence pylint bug
        for k, v in torch.load(args.base_ckpt, map_location='cpu').items():
            if k.replace('module.', '') in odict:
                odict[k.replace('module.', '')] = v
            else:
                print('Ignoring key {%s}'%k.replace('module.', ''))
        yolo.load_state_dict(odict)
        yolo.to(args.device)
        yolo.eval()

        proto = pnet(consider_coarse=not args.coarse_disabled,
                     norm=False)
        proto.to(args.device)

        return yolo, proto

    if args.model == 'residual':
        yolo = YoloVecs(nc=31,
                        use_gt_boxes=True,
                        use_fcn=args.use_fcn,
                        is_base=False,
                        mask_extra_coarse=args.mask_extra,
                        dataset=dataset,
                        pool_mode=args.pool_mode)
        odict = yolo.state_dict()
        for k, v in torch.load(args.ckpt, map_location='cpu').items():
            if k.replace('module.', '') in odict:
                odict[k.replace('module.', '')] = v
            else:
                print('Ignoring key {%s}'%k.replace('module.', ''))
        yolo.load_state_dict(odict)
        yolo.to(args.device)
        yolo.eval()

        proto = pnet(consider_coarse=not args.coarse_disabled,
                     norm=False)
        proto.to(args.device)

        return yolo, proto

    if args.model == 'dino':
        yolo = YoloFeats(nc=31,
                         is_base=True)
        odict = yolo.state_dict()
        for k, v in torch.load(args.base_ckpt, map_location='cpu').items():
            if k.replace('module.', '') in odict:
                odict[k.replace('module.', '')] = v
            else:
                print('Ignoring key {%s}'%k.replace('module.', ''))
        yolo.load_state_dict(odict)
        yolo.to(args.device)
        yolo.eval()

        dino = DinoVecs(yolo,
                        use_gt_boxes=True)
        dino.to(args.device)
        dino.eval()

        proto = pnet(consider_coarse=not args.coarse_disabled,
                     norm=False)
        proto.to(args.device)

        return dino, proto

    if args.model == 'demo':
        yolo = YoloFeats(nc=31,
                         is_base=True)
        odict = yolo.state_dict()
        for k, v in torch.load(args.base_ckpt, map_location='cpu').items():
            if k.replace('module.', '') in odict:
                odict[k.replace('module.', '')] = v
            else:
                print('Ignoring key {%s}'%k.replace('module.', ''))
        yolo.load_state_dict(odict)
        yolo.to(args.device)
        yolo.eval()

        demo = DemoVecs(yolo,
                        use_gt_boxes=True)
        demo.to(args.device)
        demo.eval()

        proto = pnet(consider_coarse=not args.coarse_disabled,
                     norm=False)
        proto.to(args.device)

        return demo, proto
