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


from copy import deepcopy
import json
import yaml

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG

class PerSegDataset(YOLODataset):
    """
        perseg dataset
    """
    def __init__(self,
                 data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['perseg'],
                 coarse_labels=True,
                 val=False,
                 augment=True,
                 path="datasets/perseg.yaml",
                 use_segments=False,
                 use_keypoints=False,
                 **kwargs):

        img_path = data_path
        augment = True and augment

        with open(path, encoding='utf-8') as fin:
            data = yaml.load(fin, yaml.BaseLoader)
        data['names'] = data['fine_names']

        self.data_path = path
        self.val = val

        if coarse_labels:
            self.names = data['coarse_names']
        else:
            self.names = data['fine_names']

        self.nc = len(self.names)
        self.idmap = data['idmap']
        self.coarse_labels = coarse_labels
        self.valid_coarse = sorted([int(i) for i in set(self.idmap.values())])

        hyp = DEFAULT_CFG
        hyp.copy_paste = .5
        super().__init__(img_path=img_path, data=data, use_segments=use_segments, use_keypoints=use_keypoints, hyp=hyp, augment=augment, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.coarse_labels:
            for i in range(item['cls'].shape[0]):
                item['cls'][i] = int(self.idmap[str(item['cls'][i].int().item())])
        return item

class EpisodicPerSeg(PerSegDataset):
    """
        episodic perseg dataset
    """
    def __init__(self,
                 data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['perseg'],
                 coarse_labels=True,
                 val=False,
                 augment=True,
                 path="datasets/perseg.yaml",
                 use_segments=False,
                 use_keypoints=False,
                 cache_dataset=False,
                 **kwargs):
        super().__init__(data_path, coarse_labels, val, augment, path, use_segments, use_keypoints, **kwargs)
        self.val = val

        self.cache_dataset = cache_dataset
        if self.cache_dataset:
            self.cache = {}

        with open(data_path+'/episodes.json', encoding='utf-8') as fin:
            self.episodes = json.load(fin)
        self.epid = -1
        self.init_episode()

    def init_episode(self, epid=None):
        """
            change epidsode id
        """
        if epid is None:
            self.epid = (self.epid + 1) % len(self.episodes)
        else:
            self.epid = epid % len(self.episodes)

        fnames = [f.replace('\\', '/').split('/')[-1] for f in self.im_files]
        epfiles = self.episodes[self.epid]['val' if self.val else 'train']
        self.vids = [fnames.index(f) for f in epfiles]
        self.len = len(self.vids)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = self.vids[index]

        if self.cache_dataset and index in self.cache:
            return deepcopy(self.cache[index])

        item = super().__getitem__(index)

        if self.cache_dataset:
            self.cache[index] = deepcopy(item)

        return item
