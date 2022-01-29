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

from yacs.config import CfgNode as CN

_C = CN()

# ------------------------------------------------------------------------------
_C.DEVICE = 'cuda:0'

# ------------------------------------------------------------------------------
_C.DATASET = CN()

_C.DATASET.ROOT_PATH = './data'
_C.DATASET.DOWNLOAD = True

# ------------------------------------------------------------------------------
_C.DATASET.AUG = CN()

_C.DATASET.AUG.SIZE = (224, 224)  # Height, width

_C.DATASET.AUG.NORM_MEAN = (0.485, 0.456, 0.406)
_C.DATASET.AUG.NORM_STD = (0.229, 0.224, 0.225)

_C.DATASET.AUG.HORIZONTAL_FLIP = 0.5

# Color jitter options.
_C.DATASET.AUG.BRIGHTNESS = 0.1
_C.DATASET.AUG.CONTRAST = 0.1
_C.DATASET.AUG.SATURATION = 0.1
_C.DATASET.AUG.HUE = 0.1

# ------------------------------------------------------------------------------
_C.DATA_LOADER = CN()

_C.DATA_LOADER.BATCH_SIZE = 64
_C.DATA_LOADER.SHUFFLE = True
_C.DATA_LOADER.N_WORKERS = 4

# ------------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.BACKBONE = 'resnet18'
_C.MODEL.PRETRAINED = True
_C.MODEL.N_CLASSES = 100

# ------------------------------------------------------------------------------
_C.OPTIM = CN()

_C.OPTIM.BASE_LR = 0.0005
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.WEIGHT_DECAY = 0.00005

# ------------------------------------------------------------------------------
_C.LR_SCHED = CN()

_C.LR_SCHED.STEP_SIZE = 2
_C.LR_SCHED.GAMMA = 0.7

# ------------------------------------------------------------------------------
_C.TRAIN = CN()

_C.TRAIN.N_EPOCHS = 10
_C.TRAIN.EVAL_FREQ = 1
_C.TRAIN.PRINT_FREQ = 10
_C.TRAIN.CHECKPOINT_SAVE_FREQ = 1

# ------------------------------------------------------------------------------
cfg = _C
