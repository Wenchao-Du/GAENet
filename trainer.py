#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   trainer.py
@Time    :   2021/09/25
@Author  :   Garified Du
@Version :   1.0
@License :   Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
@Desc    :   construct the training container
'''

# here put the import lib
import os
import sys
from tqdm import tqdm
from datetime import datetime
import random
import glob
import numpy as np
import time
import PIL.Image as Image
# import torchlib
import torch
import torch.nn as nn
import torch.nn.functional as tf
from torch.utils.data import DataLoader, Subset
# import kernel model
from libs.model import LFNet
from libs.loss import define_loss
from dataloaders.kitti_loader import KittiDepth
from nyuv2_dataset.nyu_dataloader import NYUDataset
from nyuv2_dataset.dense_to_sparse import UniformSampling
from nyuv2_dataset.metrics import AverageMeter as avger
from Utils.utils import define_optim, define_scheduler, Logger, AverageMeter
from libs.benchmark_metrics import Metrics
import torch.distributed as DIST
import torch.utils.data.distributed as DATA_DIST
from torch.nn.parallel import DistributedDataParallel as DDP


def set_randseed(config, local_rank):
    torch.manual_seed(config['seed'] + local_rank)
    torch.cuda.manual_seed_all(config['seed'] + local_rank)
    np.random.seed(config['seed'] + local_rank)
    random.seed(config['seed'] + local_rank)


def reduce_loss(loss):
    avgloss = loss.clone()
    DIST.all_reduce(avgloss)
    avgloss = avgloss / DIST.get_world_size()
    return avgloss


def train(config, local_rank):
    set_randseed(config, local_rank)
    DIST.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=config['gpus'],
                            rank=local_rank)
    torch.cuda.set_device(local_rank)
    if config['dataset'] == 'kitti':
        trainset = KittiDepth('train', config)
        valset = KittiDepth('selval', config)
    elif config['dataset'] == 'nyuv2':
        trainset = NYUDataset(
            os.path.join(config['data_root'], 'train'), 'train',
            UniformSampling(config['samples'], config['max_depth']))
        valset = NYUDataset(
            os.path.join(config['data_root'], 'val'), 'val',
            UniformSampling(config['nsp'], config['max_depth']))
    else:
        raise Exception('the dataset does not exist!!')
    model = LFNet(config)
    optimizer = define_optim(config['optimizer'], model.parameters(),
                             float(config['lr']), 0)

    criterion = define_loss(config['loss_type'])
    scheduler = define_scheduler(optimizer, config)

    if local_rank == 0:
        save_folder = os.path.join(
            config['save_root'],
            '{}_{}_{}_batch_{}_lr_{}_nsp_{}_knn_{}'.format(
                datetime.today().year,
                datetime.today().month,
                datetime.today().day, config['batch_size'], config['lr'],
                config['nsp'], config['knn']))
        print('=>Save folder: {}\n'.format(save_folder))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    best_epoch = 0
    lowest_loss = np.inf
    resume = 0
    if config['resume'] != 'None':
        checkpoint = torch.load(config['resume'],
                                map_location=torch.device('cpu'))
        resume = checkpoint['epoch']
        lowest_loss = checkpoint['loss']
        best_epoch = checkpoint['best epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to('cuda')
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})".format(
            resume, checkpoint['epoch']))
    log_file = 'log_train_start_{}.txt'.format(resume)
    if local_rank == 0:
        sys.stdout = Logger(os.path.join(save_folder, log_file))
        print(config)
        print("Number of parameters in model is {:.3f}M".format(
            sum(tensor.numel() for tensor in model.parameters()) / 1e6))
    model.cuda(local_rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True)
    train_sampler = DATA_DIST.DistributedSampler(trainset,
                                                 num_replicas=config['gpus'],
                                                 rank=local_rank)
    val_sampler = DATA_DIST.DistributedSampler(valset,
                                               num_replicas=config['gpus'],
                                               rank=local_rank)
    train_loader = DataLoader(trainset,
                              batch_size=config['batch_size'],
                              shuffle=False,
                              num_workers=config['threads'],
                              pin_memory=True,
                              sampler=train_sampler,
                              drop_last=True)
    
    val_loader = DataLoader(valset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            sampler=val_sampler)
    for epoch in range(resume, config['epoches']):
        print('=> Starch Epoch {}\n'.format(epoch))
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        if config['lr_policy'] == 'plateau':
            print('learning rate is set to {}.\n'.format(
                optimizer.param_groups[0]['lr']))
        train_sampler.set_epoch(epoch)
        batch_time = AverageMeter()
        losses = AverageMeter()
        metric_train = Metrics()
        rmse_train = AverageMeter()
        mae_train = AverageMeter()

        time_snap = time.time()
        for i, inputs in tqdm(enumerate(train_loader)):
            image, lidars, gt = inputs['rgb'].cuda(
                non_blocking=True), inputs['pc'].cuda(
                    non_blocking=True), inputs['gt'].cuda(non_blocking=True)
            output = model(image, lidars)
            loss = criterion(output, gt)
            if config['clip_grad_norm'] != 0:
                nn.utils.clip_grad_norm_(model.parameters(),
                                         config['clip_grad_norm'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r_loss = reduce_loss(loss)
            losses.update(r_loss.item(), image.size(0))
            metric_train.calculate(output.detach(), gt)
            rmse_train.update(metric_train.get_metric('rmse'),
                              metric_train.num)
            mae_train.update(metric_train.get_metric('mae'), metric_train.num)

            batch_time.update(time.time() - time_snap)
            time_snap = time.time()
            if (i + 1) % config['print_freq'] == 0:
                if local_rank == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Metric {rmse_train.val:.4f} ({rmse_train.avg:.4f})'.
                          format(epoch + 1,
                                 i + 1,
                                 len(train_loader),
                                 batch_time=batch_time,
                                 loss=losses,
                                 rmse_train=rmse_train))
            if (i + 1) % config['save_freq'] == 0:
                if local_rank == 0:
                    print('=> Start sub-selection validation set')
                rmse, mae = val(model, val_loader, config, local_rank)
                model.train()
                if local_rank == 0:
                    print("===> Average RMSE score on selection set is {:.4f}".
                          format(rmse))
                    print("===> Average MAE score on selection set is {:.4f}".
                          format(mae))
                    print(
                        "===> Last best score was RMSE of {:.4f} in epoch {}".
                        format(lowest_loss, best_epoch))
                if rmse < lowest_loss:
                    lowest_loss = rmse
                    best_epoch = epoch
                    model_state = model.module.state_dict()
                    if local_rank == 0:
                        save_checkpoints(
                            {
                                'epoch': epoch,
                                'best epoch': best_epoch,
                                'state_dict': model_state,
                                'loss': lowest_loss,
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict()
                            }, save_folder, epoch, True)
        if local_rank == 0:
            print('=> Start selection validation set')
        rmse, mae = val(model, val_loader, config, local_rank)
        model.train()
        if local_rank == 0:
            print("===> Average RMSE score on selection set is {:.4f}".format(
                rmse))
            print("===> Average MAE score on selection set is {:.4f}".format(
                mae))
            print("===> Last best score was RMSE of {:.4f} in epoch {}".format(
                lowest_loss, best_epoch))
        is_best = False
        if local_rank == 0:
            if rmse < lowest_loss:
                is_best = True
                best_epoch = epoch
                lowest_loss = rmse
            model_state = model.module.state_dict()
            save_checkpoints(
                {
                    'epoch': epoch,
                    'best epoch': best_epoch,
                    'state_dict': model_state,
                    'loss': lowest_loss,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, save_folder, epoch, is_best)
        #  update the scheduler
        if config['lr_policy'] == 'plateau':
            scheduler.step(rmse)
        else:
            scheduler.step()


def val(model, dataloader, config, local_rank=0):
    metric = Metrics()
    rmse_stc = AverageMeter()
    mae_stc = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, inputs in tqdm(enumerate(dataloader)):
            if config['dataset'] == 'kitti':
                image, lidars, gt = inputs[0]['rgb'].cuda(local_rank), inputs[
                    0]['pc'].cuda(local_rank), inputs[0]['gt'].cuda(local_rank)
            else:
                image, lidars, gt = inputs['rgb'].cuda(
                    non_blocking=True), inputs['pc'].cuda(
                        non_blocking=True), inputs['gt'].cuda(
                            non_blocking=True)

            output = model(image, lidars, True)
            # cal the metric and update the avg_stc
            if config['dataset'] == 'kitti':
                metric.calculate(256 * output, 256 * gt)
            else:
                metric.calculate(output, gt)
            reduce_rmse = reduce_loss(metric.get_metric('rmse').data)
            reduce_mae = reduce_loss(metric.get_metric('mae').data)
            rmse_stc.update(reduce_rmse.item())
            mae_stc.update(reduce_mae.item())

            if (i + 1) % config['print_freq'] == 0:
                print('Test: [{0}/{1}]\t'
                      'Metric {rmse_stc.val:.4f} ({rmse_stc.avg:.4f})'.format(
                          i + 1, len(dataloader), rmse_stc=rmse_stc))
    model.module.updataeParamters(config['batch_size'], config['nsp'])
    return rmse_stc.avg, mae_stc.avg


def save_checkpoints(model_state, save_folder, epoch, is_best=False):
    # save the current model while removing the past model
    filepath = os.path.join(save_folder, 'checkpoint.pth')
    print('save the current model : {} \n'.format(filepath))
    torch.save(model_state, filepath)
    if is_best:
        torch.save(model_state, os.path.join(save_folder, 'model_best.pth'))
        print('Best model in epoch {} copied!! \n'.format(epoch))