#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dgn.py
@Time    :   2020/09/27
@Author  :   Garified Du
@Version :   1.0
@License :   Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
@Desc    :   the embeded the structure-aware dgnn
'''

# here put the import lib

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')  # cpu cuda

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1,
                                                               1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous(
    )  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class SharedMLP(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 transpose=False,
                 padding_mode='zeros',
                 bn=False,
                 activation_fn=None):
        super(SharedMLP, self).__init__()

        conv_fn = nn.Conv1d
        self.conv = conv_fn(in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                            padding_mode=padding_mode)
        self.batch_norm = nn.BatchNorm1d(out_channels, eps=1e-6,
                                         momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class AttentivePooling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-1))
        self.mlp = SharedMLP(in_channels,
                             out_channels,
                             bn=True,
                             activation_fn=nn.ReLU())

    def forward(self, x):
        r"""
            Forward pass

            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0, 2, 1)).permute(0, 2, 1)

        # sum over the neighbors
        # shape (B, d_in, N, 1)
        # features = torch.sum(scores * x, dim=-1, keepdim=True)
        feature = x * scores
        return self.mlp(feature)


class PointNet(nn.Module):

    def __init__(self, emb_dims, output_channels=40):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)
        self.linear1 = nn.Linear(emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class LDGCNN(nn.Module):

    def __init__(self, knn_k, output_channels=8):
        super(LDGCNN, self).__init__()
        self.k = knn_k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn = nn.BatchNorm1d(64)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d((64 + 3) * 2, 64, kernel_size=1, bias=False), self.bn2,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d((64 * 2 + 3) * 2, 64, kernel_size=1, bias=False),
            self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(
            nn.Conv2d((64 * 3 + 3) * 2, 64, kernel_size=1, bias=False),
            self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv_out = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=1, bias=False), self.bn,
            nn.LeakyReLU(negative_slope=0.2))  # src output 64
        self.strpool = AttentivePooling(64, output_channels)

    def forward(self, input):
        x = input
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, input), dim=1)
        x = get_graph_feature(x, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x2, x1, input), dim=1)
        x = get_graph_feature(x, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x3, x2, x1, input), dim=1)
        x = get_graph_feature(x, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_out(x)
        x = self.strpool(x)
        x = x.permute(0, 2, 1)
        return x


class DGCNN(nn.Module):

    def __init__(self, knn_k, output_channels=16):
        super(DGCNN, self).__init__()
        self.k = knn_k
        self.output_channels = output_channels
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn_f1 = nn.BatchNorm1d(64)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv_fuse_3 = nn.Sequential(
            nn.Conv1d(512, 64, kernel_size=1, bias=False), self.bn_f1,
            nn.LeakyReLU(negative_slope=0.2))
        self.maxpool_3 = nn.AdaptiveMaxPool1d(self.output_channels)

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)sss
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        output = self.conv_fuse_3(torch.cat((x1, x2, x3, x4), dim=1))
        output = self.maxpool_3(output.permute(0, 2, 1))

        return output


def main(args):
    # torch.cuda.empty_cache()
    model = DGCNN(9)
    print(model)
    # model = model.cuda(0)
    inputdata = torch.rand((4, 3, 6000), dtype=torch.float32)  #.cuda(0)
    output = model(inputdata)
    print(output.size())
