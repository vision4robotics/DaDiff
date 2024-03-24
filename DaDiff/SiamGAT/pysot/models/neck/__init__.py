# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.models.neck.neck import GoogLeNetAdjustLayer, Adjust_Transformer

NECKS = {
        'GoogLeNetAdjustLayer': GoogLeNetAdjustLayer,
        'Adjust_Transformer': Adjust_Transformer,
        }

def get_neck(name, **kwargs):
    return NECKS[name](**kwargs)