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

import torch
from torch import nn
from torch.nn import functional as F

def dcos(x, y):
    """
        cosine distance
    """
    d = 1 - F.cosine_similarity(x, y, dim=0) # pylint: disable=not-callable
    return d

def dl1(x, y):
    """
        l1 distance
    """
    d = F.l1_loss(x, y)
    return d

def dl2(x, y):
    """
        l2 distance
    """
    d = F.mse_loss(x, y)
    return d

class Conditional(nn.Module):
    """
        conditional protonet module
    """
    def __init__(self,
                 consider_coarse=True,
                 norm=False):
        super().__init__()

        self.consider_coarse = consider_coarse
        self.norm = norm

        self.prototypes = {}
        self.counts = {}

        self.dists = [dcos, dl1, dl2]

    def reset(self):
        """
            reset protonet
        """
        self.prototypes = {}
        self.counts = {}

    def train_protos(self, x, y):
        """
            train prototypes
        """
        for i, sample in enumerate(x):
            for f, _, cl in sample:
                if self.norm:
                    f = f / f.norm()
                fl = y[i].item()
                if not cl in self.prototypes:
                    self.prototypes[cl] = {-1: torch.zeros_like(f)} if self.consider_coarse else {}
                    self.counts[cl] = {-1: 0} if self.consider_coarse else {}
                if not fl in self.prototypes[cl]:
                    self.prototypes[cl][fl] = torch.zeros_like(f)
                    self.counts[cl][fl] = 0

                if self.consider_coarse:
                    self.prototypes[cl][-1] = (self.prototypes[cl][-1]*self.counts[cl][-1] + f)/(self.counts[cl][-1]+1)
                    self.counts[cl][-1] += 1

                self.prototypes[cl][fl] = (self.prototypes[cl][fl]*self.counts[cl][fl] + f)/(self.counts[cl][fl]+1)
                self.counts[cl][fl] += 1

    def forward(self, x, preds):
        """
            torch module forward
        """
        for xs, ps in zip(x, preds):
            for (f, _, cl), box in zip(xs, ps):
                if self.norm:
                    f = f / f.norm()
                if cl in self.prototypes:
                    cfs, protos = list(self.prototypes[cl]), list(self.prototypes[cl].values())
                    dists = torch.zeros(len(self.dists), len(protos), device=ps.device)
                    for di, dist in enumerate(self.dists):
                        for pi, p in enumerate(protos):
                            dists[di, pi] = dist(f,p)
                    probs = F.softmax(1./(dists+1e-5), dim=-1).mean(dim=0)
                    cf = cfs[probs.argmax()]
                    box[-1] = cf
                else:
                    box[-1] = -1
        return preds

    def get_protos(self):
        """
            get prototypes dictionary in output
        """
        return {cl: {cf: v.numpy().tolist() for cf, v in d.items()} for cl, d in self.prototypes.items()}

class BaseProtonet(nn.Module):
    """
        normal protonet module
    """
    def __init__(self,
                 consider_coarse=True,
                 norm=False):
        super().__init__()

        self.consider_coarse = consider_coarse
        self.norm = norm

        self.prototypes = {}
        self.counts = {}

        self.dist = dl2

    def reset(self):
        """
            reset protonet
        """
        self.prototypes = {}
        self.counts = {}

    def train_protos(self, x, y):
        """
            train prototypes
        """
        for i, sample in enumerate(x):
            for f, _, cl in sample:
                fl = y[i].item()

                if self.norm:
                    f /= f.norm()

                if fl not in self.prototypes:
                    self.prototypes[fl] = torch.zeros_like(f)
                    self.counts[fl] = 0

                if self.consider_coarse and -cl not in self.prototypes:
                    self.prototypes[-cl] = torch.zeros_like(f)
                    self.counts[-cl] = 0

                if self.consider_coarse:
                    self.prototypes[-cl] = (self.counts[-cl]*self.prototypes[-cl] + f)/(self.counts[-cl] + 1)
                    self.counts[-cl] += 1

                self.prototypes[fl] = (self.counts[fl]*self.prototypes[fl] + f)/(self.counts[fl] + 1)
                self.counts[fl] += 1

    def forward(self, x, preds):
        """
            torch module forward
        """
        fls, pts = list(self.prototypes), list(self.prototypes.values())
        ds = torch.zeros(len(pts))

        for xs, ps in zip(x, preds):
            for (f, _, _), box in zip(xs, ps):
                if self.norm:
                    f /= f.norm()

                for i, pt in enumerate(pts):
                    ds[i] = 1./(self.dist(f,pt) + 1e-5)

                box[-1] = fls[ds.argmax()]

        return preds

class SimpleShot(BaseProtonet):
    """
        simpleshot
    """
    def __init__(self, consider_coarse=True, norm=True):
        super().__init__(consider_coarse=consider_coarse, norm=True)
