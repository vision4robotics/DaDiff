# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from siamban.core.config import cfg
from siamban.models.loss import select_cross_entropy_loss, select_iou_loss
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head
from siamban.models.neck import get_neck
from runners.diffusion import Diffusion
from main import parse_args_and_config
from models.diffusion import Model

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        args, config = parse_args_and_config()
        self.runner = Diffusion(args, config)
        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)
                            
        self.diffusion = Model(config)

        #if cfg.ALIGN.ALIGN:
           # self.align = get_neck(cfg.ALIGN.TYPE,
            #                     **cfg.ALIGN.KWARGS)
            
         #   self.align_track = get_neck(cfg.ALIGN.TYPE,
          #                       **cfg.ALIGN.KWARGS)

        # build ban head
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().cuda()

        self.e_zf = nn.Parameter(torch.randn((1,256,7,7)))
        
        self.e_xf = nn.Parameter(torch.randn((1,256,31,31)))

        self.align_track = get_neck(cfg.ALIGN.TYPE,
                                 **cfg.ALIGN.KWARGS)

    def template(self, z):
        zf = self.backbone(z)
        
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        #if cfg.ALIGN.ALIGN:
         #   zf = [self.align(zf[i]) for i in range(len(zf))]
        zf_diff=zf
        for i in range(len(zf)):
                n = zf[i].size(0)
                #e_zf = torch.randn_like(zf[i])
                #n = zf.size(0)
                e_zf = self.e_zf.repeat(n,1,1,1)
                b = self.betas
                t = 1*torch.ones(1).cuda().int()
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                x = zf[i] * a.sqrt() + e_zf * (1.0 - a).sqrt()
                zf_diff[i], _ = self.runner.sample(x, zf[i], self.diffusion)
                zf_diff[i] = zf_diff[i].cuda()

        zf_diff = [self.align_track(zf_diff[i]) for i in range(len(zf_diff))]

        #if cfg.ALIGN.ALIGN:
         #   zf_diff = [self.align_track(zf_diff[i]) for i in range(len(zf_diff))]
            
        self.zf = zf_diff

    def track(self, x):
        xf = self.backbone(x)
        
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        #if cfg.ALIGN.ALIGN:
         #   xf = [self.align(xf[i]) for i in range(len(xf))]
        xf_diff=xf
        for i in range(len(xf)):
                n = xf[i].size(0)
                #e_xf = torch.randn_like(xf[i])
                e_xf = self.e_xf.repeat(n,1,1,1)
                b = self.betas
                t = 1*torch.ones(1).cuda().int()
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                x = xf[i] * a.sqrt() + e_xf * (1.0 - a).sqrt()
                xf_diff[i], _ = self.runner.sample(x, xf[i], self.diffusion)
                xf_diff[i] = xf_diff[i].cuda()
        
        xf_diff = [self.align_track(xf_diff[i]) for i in range(len(xf_diff))]
                
        
            
        cls, loc = self.head(self.zf, xf_diff)
        return {
                'cls': cls,
                'loc': loc
               }


    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        
        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        #if cfg.ALIGN.ALIGN:
         #   zf = [self.align(zf[i]) for i in range(len(zf))]
          #  xf = [self.align(xf[i]) for i in range(len(xf))]

        zf_diff=zf
        zf_times = zf_diff
        for i in range(len(zf)):
                n = zf[i].size(0)
                #e_zf = torch.randn_like(zf[i])
                e_zf = self.e_zf.repeat(n,1,1,1)
                b = self.betas.cuda()
                t = 1*torch.ones(1).cuda().int()
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                x = zf[i] * a.sqrt() + e_zf * (1.0 - a).sqrt()
                zf_diff[i], zf_times[i] = self.runner.sample(x, zf[i], self.diffusion)
                zf_diff[i] = zf_diff[i].cuda()
                zf_times[i] = zf_times[i].cuda()

        xf_diff=xf
        xf_times = xf_diff
        for i in range(len(xf)):
                n = xf[i].size(0)
                #e_xf = torch.randn_like(xf[i])
                e_xf = self.e_xf.repeat(n,1,1,1)
                b = self.betas.cuda()
                t = 1*torch.ones(1).cuda().int()
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                x = xf[i] * a.sqrt() + e_xf * (1.0 - a).sqrt()
                xf_diff[i], xf_times[i] = self.runner.sample(x, xf[i], self.diffusion)
                xf_diff[i] = xf_diff[i].cuda()
                xf_times[i] = xf_times[i].cuda()
        
        zf_diff = [self.align_track(zf_diff[i]) for i in range(len(zf_diff))]
        xf_diff = [self.align_track(xf_diff[i]) for i in range(len(xf_diff))]

        cls, loc = self.head(zf_diff, xf_diff)

        # get loss

        # cls loss with cross entropy loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)

        # loc loss with iou loss
        loc_loss = select_iou_loss(loc, label_loc, label_cls)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        return  outputs, zf_times, xf_times #zf_diff, xf_diff #, zf, xf
