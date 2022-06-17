"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
from sklearn.metrics import mean_squared_error


class Metrics(object):
    def __init__(self):
        self.rmse, self.mae = 0, 0
        self.num = 0

    def calculate(self, prediction, gt):
        valid_mask = (gt > 0).detach()
        self.num = valid_mask.sum().item()
        prediction = prediction[valid_mask]  # conver m to mm
        gt = gt[valid_mask]  # convert m to mm
        abs_diff = (prediction - gt).abs()
        self.rmse = torch.sqrt(torch.mean(torch.pow(abs_diff, 2)))
        self.mae = abs_diff.mean()

    def get_metric(self, metric_name):
        return self.__dict__[metric_name]


def allowed_metrics():
    return Metrics().__dict__.keys()
