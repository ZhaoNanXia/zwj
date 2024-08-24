import numpy as np
import torch
import torch.nn as nn


class LayerNorm:
    def __init__(self, eps=1e-5):
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)  # keepdims为True会使计算结果的维度与输入维度一致
        std = np.std(x, axis=-1, keepdims=True)

        x_normal = (x - mean) / (std + self.eps)

        return x_normal


class BatchNorm:
    def __init__(self, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        self.run_mean = None
        self.run_std = None

    def forward(self, x, training=True):
        if self.run_mean is None:
            self.run_mean = np.mean(x, axis=0)
            self.run_std = np.std(x, axis=0)

        if training:
            batch_mean = np.mean(x, axis=0)
            batch_std = np.std(x, axis=0)

            x_normal = (x - batch_mean) / np.sqrt(batch_std + self.eps)

            self.run_mean = self.momentum * self.run_mean + (1 - self.momentum) * batch_mean
            self.run_std = self.momentum * self.run_std + (1 - self.momentum) * batch_std

        else:
            x_normal = (x - self.run_mean) / np.sqrt(self.run_std + self.eps)

        return x_normal
