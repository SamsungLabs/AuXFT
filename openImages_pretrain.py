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

import random

from copy import deepcopy
from shutil import rmtree
from tqdm import tqdm
from numpy import random as npr

import torch
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from torchvision.transforms import Normalize

from datasets import OpenImages
from models import DinoFeats, YoloFeats, ReconLoss
from utils import clean_predictions, Metrics

def set_seed(seed):
    """
        sets the rng seeds for all libraries
    """
    torch.manual_seed(seed)
    npr.seed(seed)
    random.seed(seed)

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

def init_loaders_and_models(rank, world_size, args):
    """
        initialize stuff here, to reduce SAM cost
    """
    tset = OpenImages(imgsz=672)
    vset = OpenImages(imgsz=672, val=True)

    tsampler = DistributedSampler(tset, num_replicas=world_size, rank=rank, shuffle=True)
    tloader = DataLoader(tset,
                         args.batch_per_gpu,
                         num_workers=16,
                         pin_memory=True,
                         drop_last=True,
                         sampler=tsampler,
                         collate_fn=tset.collate_fn)

    vloader = DataLoader(vset,
                         args.batch_per_gpu,
                         num_workers=8,
                         pin_memory=True,
                         drop_last=False,
                         shuffle=False,
                         collate_fn=tset.collate_fn)

    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    norm.to('cuda')

    dino = DinoFeats()
    dino.to('cuda')
    dino.train()
    dino = DDP(dino, device_ids=[rank], find_unused_parameters=args.track_parameters)

    yolo = YoloFeats(nc=tset.nc, verbose=False)
    sdict = dict(yolo.state_dict()) #silence pylint bug
    ndict = torch.load('models/checkpoints/yolov8n.pt', map_location='cpu')
    if 'model' in ndict:
        ndict = ndict['model'].state_dict()
        for k in sdict:
            if 'model.22.cv3' not in k and 'fmap' not in k:
                sdict[k] = ndict[k]
    else:
        sdict = {k.replace('module.', ''): v for k,v in ndict.items()}
    yolo.load_state_dict(sdict) # initialize yolo with distilled dino weights
    yolo.to('cuda')
    yolo.train()
    yolo = DDP(yolo, device_ids=[rank], find_unused_parameters=args.track_parameters)

    ema_dict = deepcopy(yolo.state_dict())
    # freeze layers as in: https://github.com/ultralytics/
    #                      ultralytics/blob/main/ultralytics/engine/trainer.py#L214
    for n, p in yolo.named_parameters():
        if '.dfl' in n:
            p.requires_grad = False

    return vset, tloader, vloader, norm, dino, yolo, ema_dict

def init_losses_and_optim(args, yolo, dino, tloader):
    """
        init stuff here, to reduce SAM cost
    """
    rec = ReconLoss()
    det = yolo.module.init_criterion()

    param_groups = [
        {'lr': args.lr, 'params':
            [p for n, p in yolo.named_parameters()
                if p.requires_grad and 'Norm' not in n], 'weight_decay': args.wd},
        {'lr': args.lr, 'params':
            [p for n, p in yolo.named_parameters()
                if p.requires_grad and 'Norm' in n], 'weight_decay': 0},
        {'lr':  args.dino_lr_scale*args.lr, 'params':
            [p for p in dino.module.parameters() if p.requires_grad], 'weight_decay': 0}
    ]

    optim = Adam(param_groups, betas=(0.937, 0.999))
    scheduler = lr_scheduler.ChainedScheduler([
        lr_scheduler.LinearLR(optim, 1e-6, 1, args.warmup),
        lr_scheduler.CosineAnnealingLR(optim, args.epochs*len(tloader))
    ])
    return rec, det, optim, scheduler

def train_log(writer, l, it, optim, yo, box, cls,
              dfl, args, lc, lf1, lf2, lf3, dino,
              x, df, yf1, yf2, yf3):
    """
        log stuff here to reduce sam cost
    """
    writer.add_scalar('train/ltot', l.item(), it)
    writer.add_scalar('train/lr/yolo', optim.param_groups[0]['lr'], it)
    writer.add_scalar('train/wd/yolo', optim.param_groups[0]['weight_decay'], it)
    writer.add_scalar('train/lr/dino', optim.param_groups[-1]['lr'], it)

    writer.add_scalar('train/yolo/tot', yo.item(), it)
    writer.add_scalar('train/yolo/box', box.item(), it)
    writer.add_scalar('train/yolo/cls', cls.item(), it)
    writer.add_scalar('train/yolo/dfl', dfl.item(), it)

    if args.use_kd:
        writer.add_scalar("train/rgb", lc.item(), it)

        writer.add_scalar('train/distil/p1', lf1.item(), it)
        writer.add_scalar('train/distil/p2', lf2.item(), it)
        writer.add_scalar('train/distil/tot', lf3.item(), it)

        if it % 250 == 0:
            writer.add_image("train/input", x[0].cpu(), it)
            writer.add_image("train/orig",
                dino.module.color_features(df)[0].detach().cpu(), it)
            writer.add_image("train/part1",
                dino.module.color_features(yf3)[0].detach().cpu(), it)
            writer.add_image("train/part2",
                dino.module.color_features(yf3+yf2)[0].detach().cpu(), it)
            writer.add_image("train/recon",
                dino.module.color_features(
                    yf3+yf2+yf1)[0].detach().cpu(), it)
            writer.add_image("train/simil",
                F.cosine_similarity(yf3+yf2+yf1, df, dim=1)[0:1].detach().cpu(), it) # pylint: disable=not-callable

def update_ema(it, args, yolo, ema_dict):
    """
        ema
    """
    # ema step and reset yolo
    if it % args.ema_step == 0:
        sdict = dict(yolo.state_dict()) # silence error again
        for k in ema_dict:
            if 'model.22.cv3' not in k and 'fmap' not in k:
                ema_dict[k] = args.ema_rate*ema_dict[k] + (1-args.ema_rate)*sdict[k]
            else:
                ema_dict[k] = sdict[k]
        yolo.load_state_dict(ema_dict)
    return yolo, ema_dict

def set_wd(optim, args):
    """
        set weight decay based on learning rate
    """
    for pg in optim.param_groups:
        if pg['weight_decay'] > 0:
            pg['weight_decay'] = args.wd * pg['lr']/args.lr

def train_epoch(tloader, e, args, rank, optim,
                yolo, det, norm, dino, rec, scheduler,
                ema_dict, writer, it):
    """
        do stuff here to reduce SAM cost
    """
    # set the current epoch, otherwise same order will be used each time
    tloader.sampler.set_epoch(e)
    for _, sample in enumerate(tqdm(tloader, desc='Training Epoch [%03d/%03d]'%(e+1, args.epochs), disable=rank>0, ncols=150)):
        set_wd(optim, args)

        optim.zero_grad()
        x = sample['img'] / 255.
        x = x.to('cuda', dtype=torch.float32)

        # yolo input: simple normalization in 0-1
        pfeats, (yf1, yf2, yf3) = yolo(x)

        # yolo detection loss
        yo, (box, cls, dfl) = det(pfeats, sample)
        lod = yo/args.batch_per_gpu

        if args.use_kd:
            nx = norm(x)
            df, rdf, _ = dino(nx)

            lc = rec(rdf, df)

            lf1 = rec(yf3, df)
            lf2 = rec(yf3+yf2, df)
            lf3 = rec(yf3+yf2+yf1, df)
            lkd = lc + lf1 + lf2 + lf3

            # aggregate losses
            l = lod + lkd
        else:
            l = lod

        l.backward()

        if rank == 0:
            train_log(writer, l, it, optim, yo, box, cls,
                        dfl, args, lc, lf1, lf2, lf3, dino,
                        x, df, yf1, yf2, yf3)

        optim.step()
        scheduler.step()
        it += 1

        yolo, ema_dict = update_ema(it, args, yolo, ema_dict)
    return it

def eval_epoch(yolo, args, dino, vset, vloader, e, rank, norm, rec, det, writer):
    """
        do stuff here to reduce SAM cost
    """
    torch.save(yolo.state_dict(), args.logdir+'/yolo_latest.pth')
    torch.save(dino.module.colmap.state_dict(), args.logdir+'/colmap.pth')

    metrics = Metrics(vset.names, conf=0.001)
    ayo, abox, acls, adfl = 0, 0, 0, 0
    alc, alf1, alf2, alf3 = 0, 0, 0, 0
    with torch.inference_mode():
        pbar = tqdm(vloader, desc='Validation Epoch [%03d/%03d], mAP50-90: %02.2f%%'%(e+1, args.epochs, 0), disable=rank>0, ncols=150)
        for _, sample in enumerate(pbar):
            x = sample['img'] / 255.
            x = x.to('cuda', dtype=torch.float32)

            # yolo input: simple normalization in 0-1
            (pred, pfeats), (yf1, yf2, yf3) = yolo(x)

            boxes = yolo.module.get_results(pred)
            for i, box in enumerate(boxes):
                box, labels, cls = clean_predictions(box, sample, i)
                metrics(box, labels, cls)

            if args.use_kd:
                nx = norm(x)
                df, rdf, _ = dino(nx)

                alc += rec(rdf, df)

                alf1 += rec(yf3, df)
                alf2 += rec(yf3+yf2, df)
                alf3 += rec(yf3+yf2+yf1, df)

            yo, (box, cls, dfl) = det(pfeats, sample)
            ayo += yo
            abox += box
            acls += cls
            adfl += dfl

            map50, map75, map50_95 = metrics.get_ap()
            pbar.set_description('Validation Epoch [%03d/%03d], mAP50-90: %02.2f%%'%(e+1, args.epochs, map50_95))

        if args.use_kd:
            writer.add_scalar("val/rgb", alc.item()/len(vloader), e+1)

            writer.add_scalar('val/distil/p1', alf1.item()/len(vloader), e+1)
            writer.add_scalar('val/distil/p2', alf2.item()/len(vloader), e+1)
            writer.add_scalar('val/distil/tot', alf3.item()/len(vloader), e+1)

        writer.add_scalar('val/yolo/tot', ayo.item()/len(vloader), e+1)
        writer.add_scalar('val/yolo/box', abox.item()/len(vloader), e+1)
        writer.add_scalar('val/yolo/cls', acls.item()/len(vloader), e+1)
        writer.add_scalar('val/yolo/dfl', adfl.item()/len(vloader), e+1)

        writer.add_scalar('val/metrics/mAP50', map50, e+1)
        writer.add_scalar('val/metrics/mAP75', map75, e+1)
        writer.add_scalar('val/metrics/mAP50-95', map50_95, e+1)

    return map50_95

def main(rank, world_size, args):
    """
        main function, as required by DDP
    """

    dist_url = "env://"

    # select the correct cuda device
    torch.cuda.set_device(rank)

    # initialize the process group
    print(f"| distributed init (rank {rank}): {dist_url}", flush=True)
    dist.init_process_group("nccl",
                            rank=rank,
                            init_method=dist_url,
                            world_size=world_size)
    dist.barrier()

    set_seed(args.seed)

    if rank == 0:
        rmtree(args.logdir, ignore_errors=True)
        writer = SummaryWriter(args.logdir, flush_secs=0.5)

        # extra initializations that need first run on master process
        _ = OpenImages(imgsz=672)
        _ = OpenImages(imgsz=672, val=True)
        dino = DinoFeats()
    else:
        writer = None

    dist.barrier()

    vset, tloader, vloader, norm, dino, yolo, ema_dict = init_loaders_and_models(rank, world_size, args)
    rec, det, optim, scheduler = init_losses_and_optim(args, yolo, dino, tloader)

    bap = 0
    it = 0
    for e in range(args.epochs):
        dino.train()
        yolo.train()

        it = train_epoch(tloader, e, args, rank, optim, yolo, det, norm, dino, rec, scheduler, ema_dict, writer, it)

        dist.barrier()

        dino.eval()
        yolo.eval()
        if rank == 0:

            map50_95 = eval_epoch(yolo, args, dino, vset, vloader, e, rank, norm, rec, det, writer)

            if bap < map50_95:
                bap = map50_95
                torch.save(yolo.state_dict(), args.logdir+'/yolo_best.pth')

    if rank == 0:
        torch.save(yolo.state_dict(), args.logdir+'/yolo_final.pth')
        torch.save(dino.module.colmap.state_dict(), args.logdir+'/colmap.pth')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_per_gpu", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--ema_step", type=int, default=10)
    parser.add_argument("--ema_rate", type=float, default=.6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dino_lr_scale", type=float, default=2)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--use_kd", type=str2bool, default=True)
    parser.add_argument('--track_parameters', action='store_true')
    parser.add_argument("--logdir", type=str, default="logs/OI")
    g_args = parser.parse_args()

    import os
    main(int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), g_args)
