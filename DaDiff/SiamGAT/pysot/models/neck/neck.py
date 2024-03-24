# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from pysot.models.neck.tran import Transformer


class GoogLeNetAdjustLayer(nn.Module):
    '''
    with mask: F.interpolate
    '''
    def __init__(self, in_channels, out_channels, crop_pad=0, kernel=1):
        super(GoogLeNetAdjustLayer, self).__init__()
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel),
            nn.BatchNorm2d(out_channels, eps=0.001),
        )
        self.crop_pad = crop_pad

    def forward(self, x):
        x = self.channel_reduce(x)

        if x.shape[-1] > 25 and self.crop_pad > 0:
            crop_pad = self.crop_pad
            x = x[:, :, crop_pad:-crop_pad, crop_pad:-crop_pad]

        return x

class Adjust_Transformer(nn.Module):
    def __init__(self, channels=256):
        super(Adjust_Transformer, self).__init__()

        self.row_embed = nn.Embedding(50, channels//2)
        self.col_embed = nn.Embedding(50, channels//2)
        self.reset_parameters()

        self.transformer = Transformer(channels, nhead = 8, num_encoder_layers = 1, num_decoder_layers = 0)

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x_f):
        # adjust search features
        h, w = x_f.shape[-2:]
        i = torch.arange(w).cuda()
        j = torch.arange(h).cuda()
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
            ], dim= -1).permute(2, 0, 1).unsqueeze(0).repeat(x_f.shape[0], 1, 1, 1)
        b, c, w, h = x_f.size()
        x_f = self.transformer((pos+x_f).view(b, c, -1).permute(2, 0, 1),\
                                (pos+x_f).view(b, c, -1).permute(2, 0, 1),\
                                    (pos+x_f).view(b, c, -1).permute(2, 0, 1))
        x_f = x_f.permute(1, 2, 0).view(b, c, w, h)

        return x_f
