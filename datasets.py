#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Milan Ondrašovič <milan.ondrasovic@gmail.com>
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


def make_transform(cfg, *, is_train=True):
    transform = []

    if is_train:
        color_jitter_transform = transforms.ColorJitter(
            brightness=cfg.DATASET.AUG.BRIGHTNESS,
            contrast=cfg.DATASET.AUG.CONTRAST,
            saturation=cfg.DATASET.AUG.SATURATION,
            hue=cfg.DATASET.AUG.HUE
        )
        transform.append(color_jitter_transform)

        horizontal_flip = cfg.DATASET.AUG.HORIZONTAL_FLIP
        if horizontal_flip:
            horizontal_flip_transform = transforms.RandomHorizontalFlip(
                horizontal_flip
            )
            transform.append(horizontal_flip_transform)

    resize_transform = transforms.Resize(
        size=cfg.DATASET.AUG.SIZE, interpolation=Image.BICUBIC
    )
    transform.append(resize_transform)

    to_tensor_transform = transforms.ToTensor()
    transform.append(to_tensor_transform)

    normalize_transform = transforms.Normalize(
        mean=cfg.DATASET.AUG.NORM_MEAN, std=cfg.DATASET.AUG.NORM_STD
    )
    transform.append(normalize_transform)

    transform = transforms.Compose(transform)

    return transform


def make_dataset(cfg, transform, *, is_train=True):
    dataset = datasets.CIFAR100(
        root=cfg.DATASET.ROOT_PATH,
        train=is_train,
        transform=transform,
        download=cfg.DATASET.DOWNLOAD
    )
    return dataset


def make_data_loader(cfg, dataset, sampler=None, *, is_train=True):
    # If sampler is provided, then the shuffle parameter has to be avoided.
    shuffle = is_train if sampler is None else None
    pin_memory = torch.cuda.is_available()

    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=cfg.DATA_LOADER.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.N_WORKERS,
        pin_memory=pin_memory,
        drop_last=True
    )

    return data_loader


def make_data_loader_pack(cfg, *, is_train=True, is_distributed=False):
    transform = make_transform(cfg, is_train=is_train)
    dataset = make_dataset(cfg, transform, is_train=is_train)

    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train)
    else:
        sampler = None
    data_loader = make_data_loader(cfg, dataset, sampler, is_train=is_train)

    return data_loader
