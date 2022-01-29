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
import tqdm

from utils import save_checkpoint


def _process_one_epoch(
    model,
    criterion,
    optimizer,
    lr_scheduler,
    data_loader,
    device,
    epoch,
    n_epochs,
    print_freq,
    *,
    is_train=True,
):
    if is_train:
        model.train()
    else:
        model.eval()

    phase = 'TRAIN' if is_train else 'TEST'
    desc_head = f"[{phase}] epoch: {epoch}/{n_epochs}"

    running_loss_sum = 0.0
    running_correct_sum = 0
    n_total_samples = 0

    pbar = tqdm.tqdm(enumerate(data_loader, start=1), total=len(data_loader))
    for batch_num, (images, labels) in pbar:
        n_total_samples += len(images)

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            _, preds = torch.max(outputs, dim=1)

        curr_loss = loss.item()
        running_loss_sum += curr_loss
        n_curr_correct = torch.sum(preds == labels)
        running_correct_sum += n_curr_correct

        if (batch_num % print_freq) == 0:
            running_loss = running_loss_sum / batch_num
            curr_accuracy = n_curr_correct / len(images)
            running_accuracy = running_correct_sum / n_total_samples
            curr_desc = (
                f"{desc_head} | "
                f"loss: {curr_loss:.4f} ({running_loss:.4f}) | "
                f"accuracy: {curr_accuracy:6.2%} ({running_accuracy:6.2%})"
            )
            pbar.set_description(curr_desc)
            pbar.update(n=print_freq)

    if is_train:
        lr_scheduler.step()


def do_train(
    model,
    criterion,
    optimizer,
    lr_scheduler,
    data_loader_tr,
    device,
    n_epochs,
    start_epoch=1,
    data_loader_va=None,
    eval_freq=1,
    checkpoints_dir_path=None,
    checkpoint_save_freq=None,
    print_freq=10
):
    for epoch in range(start_epoch, n_epochs + 1):
        _process_one_epoch(
            model,
            criterion,
            optimizer,
            lr_scheduler,
            data_loader_tr,
            device,
            epoch,
            n_epochs,
            print_freq,
            is_train=True
        )

        if checkpoints_dir_path and ((epoch % checkpoint_save_freq) == 0):
            save_checkpoint(
                checkpoints_dir_path, model, optimizer, lr_scheduler, epoch
            )

        if (data_loader_va is not None) and ((epoch % eval_freq) == 0):
            _process_one_epoch(
                model,
                criterion,
                optimizer,
                lr_scheduler,
                data_loader_va,
                device,
                epoch,
                n_epochs,
                print_freq,
                is_train=False
            )
