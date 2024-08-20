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

from torch.nn import CosineSimilarity, Module, MSELoss, L1Loss

class ReconLoss(Module):
    """
        reconstruction loss
    """
    def __init__(self):
        super().__init__()

        self.mse = MSELoss()
        self.l1 = L1Loss()

    def forward(self, x, y):
        """
            torch module forward
        """
        return self.mse(x,y) + self.l1(x, y)

class CosineLoss(Module):
    """
        cosine loss
    """
    def __init__(self):
        super().__init__()
        self.sim = CosineSimilarity()

    def forward(self, x, y):
        """
            torch module forward
        """
        l = 1. - self.sim(x, y)
        return l.mean()
