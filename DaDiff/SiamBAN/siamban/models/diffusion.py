# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
#import scipy.signal
import torch
#from torch_utils import persistence
#from torch_utils import misc
#from torch_utils.ops import upfirdn2d
#from torch_utils.ops import grid_sample_gradfix
#from torch_utils.ops import conv2d_gradfix

#from training.diffaug import DiffAugment
#from training.adaaug import AdaAugment

#----------------------------------------------------------------------------
# Helpers for doing defusion process.

#@persistence.persistent_class
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

class Diffusion(torch.nn.Module):
    def __init__(self,
        beta_schedule='linear', beta_start=1e-4, beta_end=2e-2,
        t_min=10, t_max=100, noise_std=0.05,
        aug='no', ada_maxp=None, ts_dist='priority',
    ):
        super().__init__()
        self.p = 0.0       # Overall multiplier for augmentation probability.

    def forward(self, x0):
        betas = get_beta_schedule(
            beta_schedule= 'linear', #config.diffusion.beta_schedule,
            beta_start= 0.0001, #config.diffusion.beta_start,
            beta_end= 0.02, #config.diffusion.beta_end,
            num_diffusion_timesteps= 100 ) #config.diffusion.num_diffusion_timesteps,
        
        betas = torch.from_numpy(betas).float().cuda()
        
        e = torch.randn_like(x0)
        #e_xf = torch.randn_like(xf[0])
        b = betas
        n = 16
        # antithetic sampling
        t = torch.randint(
                    low=0, high=100, size=(n // 2 + 1,)
                ).cuda()
        t = torch.cat([t, 100 - t - 1], dim=0)[:n]
        #diffusion
        
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
        return x

#----------------------------------------------------------------------------