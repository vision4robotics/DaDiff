# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
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

class Graph_Attention_Union(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Graph_Attention_Union, self).__init__()

        # search region nodes linear transformation
        self.support = nn.Conv2d(in_channel, in_channel, 1, 1)

        # target template nodes linear transformation
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1)

        # linear transformation for message passing
        self.g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        # aggregated feature
        self.fi = nn.Sequential(
            nn.Conv2d(in_channel*2, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, zf, xf):
        # linear transformation
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # linear transformation for message passing
        xf_g = self.g(xf)
        zf_g = self.g(zf)

        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_g_plain = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)

        similar = torch.matmul(xf_trans_plain, zf_trans_plain)
        similar = F.softmax(similar, dim=2)

        embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
        embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])

        # aggregated feature
        output = torch.cat([embedding, xf_g], 1)
        output = self.fi(output)
        return output


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        
        # load diffusion
        args, config = parse_args_and_config()
        self.runner = Diffusion(args, config)

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)
        
        # build diffusion layer
        self.diffusion = Model(config)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().cuda()

        self.e_zf = nn.Parameter(torch.randn((1,256,13,13)))
        
        self.e_xf = nn.Parameter(torch.randn((1,256,25,25)))

        # bulid align track layer
        self.align_track = get_neck(cfg.ALIGN.TYPE,
                                 **cfg.ALIGN.KWARGS)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.attention = Graph_Attention_Union(256, 256)

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

    def template(self, z, roi):
        zf = self.backbone(z, roi)
        zf_diff=zf
        #for i in range(len(zf)):
        n = zf.size(0)
        e_zf = self.e_zf.repeat(n,1,1,1)
        b = self.betas
        t = 1*torch.ones(1).cuda().int()
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = zf * a.sqrt() + e_zf * (1.0 - a).sqrt()
        zf_diff, _ = self.runner.sample(x, zf, self.diffusion)
        zf_diff = zf_diff.cuda()

        zf_diff = self.align_track(zf_diff)

        self.zf_diff = zf_diff

    def track(self, x):
        xf = self.backbone(x)
        
        xf_diff=xf
        #for i in range(len(xf)):
        n = xf.size(0)
        e_xf = self.e_xf.repeat(n,1,1,1)
        b = self.betas
        t = 1*torch.ones(1).cuda().int()
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = xf * a.sqrt() + e_xf * (1.0 - a).sqrt()
        xf_diff, _ = self.runner.sample(x, xf, self.diffusion)
        xf_diff = xf_diff.cuda()
        
        xf_diff = self.align_track(xf_diff)

        features = self.attention(self.zf_diff, xf_diff)

        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()
        target_box = data['target_box'].cuda()
        neg = data['neg'].cuda()

        # get feature
        zf = self.backbone(template, target_box)
        xf = self.backbone(search)
        
        
        zf_diff=zf
        zf_times = zf_diff
        #for i in range(len(zf)):
        n = zf.size(0)
        e_zf = self.e_zf.repeat(n,1,1,1)
        b = self.betas.cuda()
        t = 1*torch.ones(1).cuda().int()
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = zf * a.sqrt() + e_zf * (1.0 - a).sqrt()
        zf_diff, zf_times = self.runner.sample(x, zf, self.diffusion)
        zf_diff = zf_diff.cuda()
        zf_times = zf_times.cuda()        

        xf_diff=xf
        xf_times = xf_diff
        #for i in range(len(xf)):
        n = xf.size(0)
        e_xf = self.e_xf.repeat(n,1,1,1)
        b = self.betas.cuda()
        t = 1*torch.ones(1).cuda().int()
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = xf * a.sqrt() + e_xf * (1.0 - a).sqrt()
        xf_diff, xf_times = self.runner.sample(x, xf, self.diffusion)
        xf_diff = xf_diff.cuda()
        xf_times = xf_times.cuda()
        
        zf_diff = self.align_track(zf_diff) 
        xf_diff = self.align_track(xf_diff) 


        features = self.attention(zf_diff, xf_diff)

        cls, loc, cen = self.car_head(features)
        locations = compute_locations(cls, cfg.TRACK.STRIDE, cfg.TRACK.OFFSET)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc, neg
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs, zf_times, xf_times
