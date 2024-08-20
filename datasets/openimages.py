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


import yaml

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG

class OpenImages(YOLODataset):
    """
        openimages dataset
    """
    def __init__(self,
                 data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['openimages'],
                 val=False,
                 augment=True,
                 path="datasets/openimages.yaml",
                 use_segments=False,
                 use_keypoints=False,
                 **kwargs):
        if val:
            img_path = data_path + '/val'
            augment = False
        else:
            img_path = data_path + '/train'
            augment = True and augment

        with open(path, encoding='utf-8') as fin:
            data = yaml.load(fin, yaml.BaseLoader)
        self.data_path = path
        self.names = data['names']
        self.nc = len(self.names)
        hyp = DEFAULT_CFG
        hyp.copy_paste = .5
        super().__init__(img_path=img_path, data=data, use_segments=use_segments, use_keypoints=use_keypoints, hyp=hyp, augment=augment, **kwargs)
