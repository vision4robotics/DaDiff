# DaDiff: Domain-aware Diffusion Model for Nighttime UAV Tracking

[Haobo Zuo](https://scholar.google.com/citations?user=5RhJGKgAAAAJ&hl=zh-CN), [Changhong Fu](https://scholar.google.com/citations?user=zmbMZ4kAAAAJ&hl=zh-CN), [Guangze Zheng](https://scholar.google.com/citations?user=-kcZWRQAAAAJ&hl=zh-CN), [Liangliang Yao](https://vision4robotics.github.io/authors/liangliang-yao/), [Kunhan Lu](https://scholar.google.com/citations?user=aW__X-8AAAAJ&hl=zh-CN), and [Jia Pan](https://scholar.google.com/citations?hl=zh-CN&user=YYT8-7kAAAAJ). DaDiff: Domain-aware Diffusion Model for Nighttime UAV Tracking.

![featured](https://github.com/vision4snake/DaDiff/blob/main/img/featured.svg)

## Overview

**DaDiff** is a Diffusion model-based domain adaptation framework for visual object tracking. This repo contains its Python implementation.

## Testing DaDiff

### 1. Preprocessing

Before training, we need to preprocess the unlabelled training data to generate training pairs.

1. Download the nighttime train dataset [NAT2021-*train* set](https://vision4robotics.github.io/NAT2021/).

2. Follow the preprocessing of [UDAT](https://github.com/vision4robotics/UDAT) to prepare the nighttime train dataset.

3. Download the proposed [NUT-LR](https://drive.google.com/file/d/1yk7YtLW4iHfbCP_Ni0Yf6ajLsuXpM3YC/view?usp=sharing) for low-resolution object nighttime UAV tracking.

### 2. Train

Take DaDiff-GAT for instance.

1. Apart from the above target domain dataset NAT2021, you need to download and prepare source domain datasets [VID](https://image-net.org/challenges/LSVRC/2017/) and [GOT-10K](http://got-10k.aitestunion.com/downloads).

2. Download the pre-trained daytime model ([SiamGAT](https://drive.google.com/file/d/1LKU6DuOzmLGJr-LYm4yXciJwIizbV_Zf/view)/[SiamBAN](https://drive.google.com/drive/folders/17Uz3dZFOtx-uU7J4t48_nAfPXvNsQAAq?usp=sharing)) and place it at `DaDiff/SiamGAT/snapshot`.

3. Start training

   ``` python
   cd DaDiff/SiamGAT
   export PYTHONPATH=$PWD
   python tools/train.py
   ```

### 3. Test
Take DaDiff-GAT for instance.
1. For quick test, you can download our trained model for [UDAT-CAR](https://drive.google.com/file/d/1DccbQ4nh2rlni8RVykTNzuHXJgSvNE4G/view?usp=sharing) (or [UDAT-BAN](https://drive.google.com/file/d/1nKyzA0ohOmrvSvypM-0cCvGNo93ZvdLp/view?usp=sharing)) and place it at `DaDiff/SiamGAT/snapshot`.

2. Start testing

    ```python
    python tools/test.py --dataset NUT-L
    ```

### 4. Eval

1. Start evaluating
    ``` python
    python tools/eval.py --dataset NUT-L
    ```

## Demo
[![Demo Video](https://i.ytimg.com/vi/zAAx3bCElsw/maxresdefault.jpg)](https://youtu.be/zAAx3bCElsw)


### Acknowledgments

We sincerely thank the contribution of the following repos: [DDIM](https://github.com/ermongroup/ddim), [SiamGAT](https://github.com/ohhhyeahhh/SiamCAR), [SiamBAN](https://github.com/hqucv/siamban), and [UDAT](https://github.com/vision4robotics/UDAT).



### Contact

The official code of DaDiff will continue to be regularly refined and improved to ensure its quality and functionality. If you have any questions, please contact Haobo Zuo at [haobozuo@connect.hku.hk](mailto:haobozuo@connect.hku.hk) or Changhong Fu at [changhongfu@tongji.edu.cn](mailto:changhongfu@tongji.edu.cn).
