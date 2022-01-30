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

import argparse
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import cfg
from datasets import make_data_loader_pack
from losses import make_loss
from models import make_model
from optim import make_optimizer, make_lr_scheduler
from trainer import do_train
from utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Faster RCNN, Context RCNN: training/evaluation script.",
    )

    parser.add_argument(
        '-c', '--config', help="path to the YAML configuration file"
    )
    parser.add_argument(
        '--checkpoints-dir',
        help="path to the directory containing checkpoints"
    )
    parser.add_argument(
        '--log-dir',
        default='./logs',
        help="Directory containing logs, "
        "tensorboard writer data or test evaluations"
    )
    parser.add_argument(
        '--checkpoint-file',
        help="specific checkpoint file to restore the model training or "
        "evaluating from"
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help="executes model evaluation from a given checkpoint file"
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        metavar='N',
        help="local process rank"
    )
    parser.add_argument(
        'opts',
        metavar='OPTIONS',
        nargs=argparse.REMAINDER,
        help="overwriting the default YAML configuration"
    )

    args = parser.parse_args()

    return args


def ddp_setup(rank):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'

    dist.init_process_group(backend='nccl', init_method='env://', rank=rank)


def ddp_cleanup():
    dist.destroy_process_group()


def main():
    args = parse_args()

    if args.config:
        cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    local_rank = args.local_rank

    ddp_setup(rank=local_rank)

    # device = torch.device(cfg.DEVICE)
    device = torch.device(local_rank)
    torch.cuda.set_device(local_rank)

    model = make_model(cfg).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    data_loader_te = make_data_loader_pack(
        cfg, is_train=False, is_distributed=True
    )
    criterion = make_loss()
    optimizer = make_optimizer(cfg, model)
    lr_scheduler = make_lr_scheduler(cfg, optimizer)
    checkpoint_file_path = args.checkpoint_file
    if checkpoint_file_path:
        start_epoch = load_checkpoint(
            checkpoint_file_path, model, optimizer, lr_scheduler
        )
    else:
        start_epoch = 1

    if args.test_only:
        raise NotImplementedError
    else:
        n_epochs = cfg.TRAIN.N_EPOCHS
        eval_freq = cfg.TRAIN.EVAL_FREQ
        checkpoint_save_freq = cfg.TRAIN.CHECKPOINT_SAVE_FREQ
        print_freq = cfg.TRAIN.PRINT_FREQ

        data_loader_tr = make_data_loader_pack(
            cfg, is_train=True, is_distributed=True
        )

        do_train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            data_loader_tr=data_loader_tr,
            device=device,
            n_epochs=n_epochs,
            start_epoch=start_epoch,
            data_loader_va=data_loader_te,
            eval_freq=eval_freq,
            checkpoints_dir_path=args.checkpoints_dir,
            checkpoint_save_freq=checkpoint_save_freq,
            print_freq=print_freq
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
