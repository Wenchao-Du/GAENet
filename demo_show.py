#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   demo_show.py
@Time    :   2021/10/21
@Author  :   Garified Du
@Version :   1.0
@License :   Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
@Desc    :   if anything about the descriptions, please put them here. else None
'''

# here put the import lib

import os
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import h5py as h5
from nyuv2_dataset.metrics import AverageMeter as avger, Result
import torch


def metric_nyu(folder):
    filelist = os.listdir(folder)
    filelist.sort()
    avge = avger()
    for i in range(0, len(filelist)):
        filepath = os.path.join(folder, filelist[i])
        if '_gt' in filepath:
            continue
        image = np.load(filepath)
        filepath1 = filepath.replace('.npy', '_gt.npy')
        imagegt = np.load(filepath1)
        result = Result()
        output = torch.from_numpy(image)
        gt = torch.from_numpy(imagegt)
        result.evaluate(output.data, gt.data)
        avge.update(result, 0, 0, output.size(0))
    avg = avge.average()
    print(
        'mse:{}\n rmse:{}\n absrel: {}\n lg10:{} \n mae: {}\n delta1: {}\n delta2: {}\n delta3: {}\n gpu_time: {}\n data_time: {}\n'
        .format(avg.mse, avg.rmse, avg.absrel, avg.lg10, avg.mae, avg.delta1,
                avg.delta2, avg.delta3, avg.gpu_time, avg.data_time))


def static_rmse():
    WDD = [159.243996, 123.975775, 104.023267, 89.31498]
    WGAE = [158.701854, 114.861733, 92.949492, 74.45819]
    X = ['200', '500', '1000', '2000']
    fig, ax = plt.subplots()
    plt.grid()
    # plt.plot(range(300), data1, 'r', label='rmse')
    font1 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 10,
    }

    A, = plt.plot(X, WDD, 'y', label='WDD', linewidth=3, marker='o')
    B, = plt.plot(X, WGAE, 'b', label='WGAE', linewidth=3, marker='o')
    plt.legend(handles=[A, B], prop=font1)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    font3 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 15,
    }
    plt.xlabel('NSP', font3)
    plt.ylabel('RMSE(mm)', font3)
    # plt.legend()
    plt.show()


def saveim(folder, savefolder):
    if not os.path.exists(folder):
        assert ('path not exists!')
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    for file in (os.listdir(folder)):
        filepath = os.path.join(folder, file)
        image = np.load(filepath)
        savepath = os.path.join(savefolder, file)
        savepath = savepath.replace('npy', 'png')
        plt.imsave(savepath, image, cmap='jet', format='png')


def testh5(folder):
    filelist = os.listdir(folder)
    for file in filelist:
        filepath = os.path.join(folder, file)
        fp = h5.File(filepath)
        rgb = np.array(fp['rgb'])
        rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.array(fp['depth'])
        print(depth.max())


def testpgm(file):
    depthmat = Image.open(file)
    depthnp = np.array(depthmat)
    print(depthnp.max())


def showfolder(folder):
    filelist = os.listdir(folder)
    filelist.sort()
    for sp in filelist:
        image = Image.open(os.path.join(folder, sp))
        image = np.array(image)
        plt.figure('1')
        plt.imshow(image, cmap='jet')
        plt.show()


if __name__ == "__main__":
    folder = '/mnt/725AAA345AA9F54F/LFNet_Dist/checkfolder/batch_16_lr_1e-2_nsp_2000_knn_9/testset_2000'
    savefolder = '/mnt/725AAA345AA9F54F/LFNet_Dist/checkfolder/batch_16_lr_1e-2_nsp_2000_knn_9/testset_2000_results'
    saveim(folder, savefolder)
