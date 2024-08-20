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

from torch.utils.data import DataLoader
import yaml

from datasets import PerSegDataset, EpisodicPerSeg, \
                     PODDataset, EpisodicPOD, \
                     CORe50Dataset, EpisodicCORe50, \
                     iCubWorldDataset, EpisodiciCubWorld

def get_train_val_loaders(args):
    """
        parse arguments and return correct loaders
    """
    if not args.episodic:
        if args.dataset == 'pod':
            tset = PODDataset(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['pod'],
                              coarse_labels=args.coarse_labels,
                              imgsz=672,
                              val=False,
                              augment=False)
            vset = PODDataset(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['pod'],
                              coarse_labels=args.coarse_labels,
                              imgsz=672,
                              val=True,
                              val_mode=args.val_mode,
                              augment=False)
        elif args.dataset == 'perseg':
            if not args.coarse_labels:
                raise ValueError('PerSeg Dataset can only be used in episodic mode')
            tset = PerSegDataset(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['perseg'],
                                 coarse_labels=args.coarse_labels,
                                 imgsz=672,
                                 val=False,
                                 augment=False)
            vset = None
        elif args.dataset == 'core50':
            if not args.coarse_labels and not args.debug:
                raise ValueError('CORe50 Dataset can only be used in episodic mode')
            tset = CORe50Dataset(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['core50'],
                                 coarse_labels=args.coarse_labels,
                                 imgsz=672,
                                 val=False,
                                 augment=False)
            vset = None
        elif args.dataset == 'icub':
            if not args.coarse_labels:
                raise ValueError('iCubWorld Dataset can only be used in episodic mode')
            tset = iCubWorldDataset(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['icubworld'],
                                    coarse_labels=args.coarse_labels,
                                    imgsz=672,
                                    val=False,
                                    augment=False)
            vset = None
        else:
            raise ValueError('Unrecognized dataset' + str(args.dataset))
    else:
        if args.dataset == 'pod':
            tset = EpisodicPOD(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['pod'],
                               coarse_labels=False,
                               imgsz=672,
                               val=False,
                               augment=False,
                               support=args.support,
                               cache_dataset=True)
            vset = EpisodicPOD(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['pod'],
                               coarse_labels=False,
                               imgsz=672,
                               val=True,
                               val_mode=args.val_mode,
                               augment=False,
                               cache_dataset=True)
        elif args.dataset == 'perseg':
            tset = EpisodicPerSeg(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['perseg'],
                                  coarse_labels=False,
                                  imgsz=672,
                                  val=False,
                                  augment=False,
                                  cache_dataset=True)
            vset = EpisodicPerSeg(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['perseg'],
                                  coarse_labels=False,
                                  imgsz=672,
                                  val=True,
                                  augment=False,
                                  cache_dataset=True)
        elif args.dataset == 'core50':
            tset = EpisodicCORe50(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['core50'],
                                  coarse_labels=False,
                                  imgsz=672,
                                  val=False,
                                  augment=False,
                                  support=args.support,
                                  cache_dataset=True)
            vset = EpisodicCORe50(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['core50'],
                                  coarse_labels=False,
                                  imgsz=672,
                                  val=True,
                                  augment=False,
                                  support=args.support,
                                  cache_dataset=True)
        elif args.dataset == 'icub':
            tset = EpisodiciCubWorld(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['icubworld'],
                                     coarse_labels=False,
                                     imgsz=672,
                                     val=False,
                                     augment=False,
                                     support=args.support,
                                     cache_dataset=True)
            vset = EpisodiciCubWorld(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['icubworld'],
                                     coarse_labels=False,
                                     imgsz=672,
                                     val=True,
                                     augment=False,
                                     support=args.support,
                                     cache_dataset=True)
        else:
            raise ValueError('Unrecognized dataset' + str(args.dataset))

    tloader = DataLoader(tset,
                         8,
                         num_workers=0,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=False,
                         collate_fn=tset.collate_fn)
    if vset is None:
        return tloader, None

    vloader = DataLoader(vset,
                         8,
                         num_workers=0,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=False,
                         collate_fn=tset.collate_fn)

    return tloader, vloader
