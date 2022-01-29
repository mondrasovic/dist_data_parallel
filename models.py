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

from torch import nn
from torchvision import models


class Classifier(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()

        self.model = backbone
        n_features = self.model.fc.in_features
        fc_classifier = nn.Linear(n_features, n_classes)
        self.model.fc = fc_classifier

    def forward(self, x):
        return self.model(x)


def make_model(cfg):
    backbone_name = cfg.MODEL.BACKBONE
    pretrained = cfg.MODEL.PRETRAINED
    progress = False

    if backbone_name == 'resnet18':
        backbone = models.resnet18(pretrained, progress)
    else:
        raise ValueError("unsupported backbone name")

    model = Classifier(backbone, n_classes=cfg.MODEL.N_CLASSES)

    return model
