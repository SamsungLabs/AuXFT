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

from ultralytics.data.dataset import YOLODataset
import yaml

root = yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['coco']
class COCO(YOLODataset):
    """
        coco
    """
    def __init__(self,
                 val=False,
                 path="datasets/coco.yaml",
                 use_segments=False,
                 use_keypoints=False,
                 **kwargs):
        if val:
            img_path = root + '/val'
            augment = False
        else:
            img_path = root + '/train'
            augment = True

        with open(path, encoding='utf-8') as fin:
            data = yaml.load(fin, yaml.BaseLoader)
        self.names = data['names']
        self.nc = len(self.names)
        super().__init__(img_path=img_path, data=data, use_segments=use_segments, use_keypoints=use_keypoints, augment=augment, **kwargs)
