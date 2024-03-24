import os
os.environ['CUDA_VISIBLE_DEVICES']='3,1'
import logging
import time
import glob
import logging
import numpy as np
import tqdm
import torch
import argparse
#import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from models.diffusion import Model
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
#from siamban.models.model_builder import ModelBuilder
#from siamban.core.config import cfg
#from siamban.datasets.dataset import BANDataset
#from siamban.utils.distributed import dist_init, DistModule, reduce_gradients,\
 #       average_reduce, get_rank, get_world_size
#from siamban.utils.model_load import load_pretrain, restore_from
#from siamban.models.trans_discriminator import TransformerDiscriminator
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu

#logger = logging.getLogger('global')
def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


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




class Diffusion(object):
    def __init__(self, args, config, device=None):
       # cfg.merge_from_file('experiments/udatban_r50_l234/config.yaml')
       # self.model_track = ModelBuilder().cuda().eval()
       # self.model_Disc = TransformerDiscriminator(channels=256).cuda().eval()
        
        
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
 
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().cuda()#to(self.device)
        self.num_timesteps = 20 #betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
    
    def build_data_loader(self, domain):
      logger.info("build train dataset")
    # train_dataset
      #if cfg.BAN.BAN:
      train_dataset = BANDataset(domain)
      logger.info("build dataset done")

      train_sampler = None
      if get_world_size() > 1:
       train_sampler = DistributedSampler(train_dataset)
      train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)
      return train_loader
  
    def noise_estimation(self, model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
     condition = x0
     a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
     x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
     x = torch.concat([x, condition], dim=1)
     output = model(x, t.float())
     return output
     #if keepdim:
      #  return (e - output).square().sum(dim=(1, 2, 3))
     #else:
      #  return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
      
    def weightedMSE(self, D_out, label):
        # D_label = torch.FloatTensor(D_out.data.size()).fill_(1).cuda() * label.unsqueeze(1).unsqueeze(2).unsqueeze(3).cuda()
    # D_label = torch.FloatTensor(D_out.data.size()).fill_(label).cuda()
     return torch.mean((D_out - label.cuda()).abs() ** 2)

    def train(self):
        rank, world_size = dist_init()
        source_label = 0
        load_pretrain(self.model_track, cfg.TRAIN.PRETRAINED)
        load_pretrain(self.model_Disc, cfg.TRAIN.DiscPRETRAINED)
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        #dataset, test_dataset = get_dataset(args, config)
        
        target_loader = self.build_data_loader('target')
        #train_loader = data.DataLoader(
         #   dataset,
          #  batch_size=config.training.batch_size,
          #  shuffle=True,
          #  num_workers=config.data.num_workers,
        #)
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, data in enumerate(target_loader):
                zf, xf, outputs = self.model_track(data)
                n = zf[0].size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                #zf = zf.to(self.device)
                #xf = xf.to(self.device)
                #x = data_transform(self.config, x)
                e_zf = torch.randn_like(zf[0])
                e_xf = torch.randn_like(xf[0])
                b = self.betas
                
                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                
                output_zf = [self.noise_estimation(model, zf[i], t, e_zf, b) for i in range(len(zf))]
                output_xf = [self.noise_estimation(model, xf[i], t, e_xf, b) for i in range(len(xf))]
                
                interp = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
                zf_up_t = [interp(_zf) for _zf in output_zf]
                xf_up_t = [interp(_xf) for _xf in output_xf]
                D_out_z = torch.stack([self.model_Disc(F.softmax(_zf_up_t, dim=1)) for _zf_up_t in zf_up_t]).sum(0) / 3.
                D_out_x = torch.stack([self.model_Disc(F.softmax(_xf_up_t, dim=1)) for _xf_up_t in xf_up_t]).sum(0) / 3.
                D_source_label = torch.FloatTensor(D_out_z.data.size()).fill_(source_label)
                loss = 1 * (self.weightedMSE(D_out_z, D_source_label) +  self.weightedMSE(D_out_x, D_source_label)) #  / cfg.TRAIN.BATC
                
                #loss_zf = loss_registry[config.model.type](model, zf, t, e_zf, b)
                #loss_xf = loss_registry[config.model.type](model, xf, t, e_xf, b)
                
                #loss = loss_zf+loss_xf

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def sample(self, x, x0, model):
       # model = Model(self.config)

        #if not self.args.use_pretrained:
         #   print('false')
        #    if getattr(self.config.sampling, "ckpt_id", None) is None:
        #        states = torch.load(
        #            os.path.join(self.args.log_path, "ckpt.pth"),
        #            map_location=self.config.device,
        #        )
        #    else:
        #        states = torch.load(
        #            os.path.join(
        #                self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
        #            ),
         #           map_location=self.config.device,
         #       )
         #   model = model.to(self.device)
         #   model = torch.nn.DataParallel(model)
         #   model.load_state_dict(states[0], strict=True)

          #  if self.config.model.ema:
           #     ema_helper = EMAHelper(mu=self.config.model.ema_rate)
           #     ema_helper.register(model)
          #      ema_helper.load_state_dict(states[-1])
          #      ema_helper.ema(model)
          #  else:
           #     ema_helper = None
        #else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
          #  if self.config.data.dataset == "CIFAR10":
          #      name = "cifar10"
          #  elif self.config.data.dataset == "LSUN":
          #      name = f"lsun_{self.config.data.category}"
          #  elif self.config.data.dataset == "our485":
          #      name = f"lol"
          #  else:
           #     raise ValueError
            #ckpt = get_ckpt_path(f"ema_{name}")
            #print("Loading checkpoint {}".format(ckpt))
            #model.load_state_dict(torch.load(ckpt, map_location=self.device))
            #model.to(self.device)
            #model = torch.nn.DataParallel(model)

        #model.eval()

        if self.args.fid:
            x, x_all = self.sample_fid(x, x0, model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")
        #print(x.shape)
        #print(x_all.shape)
        return x, x_all

    def sample_fid(self, x, x0, model):
        config = self.config
        #img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        #print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = 1 #(total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in range(n_rounds): #tqdm.tqdm(
                #range(n_rounds), desc="Generating image samples for FID evaluation."
           # ):
                n = config.sampling.batch_size
                #x = torch.randn(
                #    n,
                #    config.data.channels,
                #    config.data.image_size,
                #    config.data.image_size,
                #    device=self.device,
               # )

                x, x_all = self.sample_image(x, x0, model)
                #x = torch.sigmoid(x)
                #x = inverse_data_transform(config, x)

                #for i in range(n):
                 #   tvu.save_image(
                  #      x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                   # )
                    #img_id += 1
        return x, x_all

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, x0, model,  last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 10

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = 20 // 2 #self.args.timesteps
                seq = range(0, 20, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, x0, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        
        x[0][-1] = x[0][-1].cuda()
        x[0][-2] = x[0][-2].cuda()
        
        x_all = torch.cat((x[0][-1],x[0][-2]),dim=0).cuda()
        #print(len(x[0]))
        #for i in range(len(x[0])-2):
         # x_all= x_all.cuda()
          #x[0][i+2] = x[0][i+2].cuda()
          #x_all = torch.cat((x_all, x[0][i+2]), dim = 0)
        if last:
            x = x[0][-1].cuda()
       # print(x_all.shape)
        #print(x.shape)
        return x, x_all

    def test(self):
        pass
