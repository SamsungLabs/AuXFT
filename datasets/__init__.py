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

from .coco import COCO
from .openimages import OpenImages
from .perseg import PerSegDataset, EpisodicPerSeg
from .poddataset import PODDataset, EpisodicPOD
from .core50dataset import CORe50Dataset, EpisodicCORe50
from .icubworlddataset import iCubWorldDataset, EpisodiciCubWorld

__all__ = [
    "COCO",
    "OpenImages",
    "PerSegDataset", "EpisodicPerSeg",
    "PODDataset", "EpisodicPOD",
    "CORe50Dataset", "EpisodicCORe50",
    "iCubWorldDataset", "EpisodiciCubWorld"
]
