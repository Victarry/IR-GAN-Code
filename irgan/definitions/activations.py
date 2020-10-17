# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Activation functions manager"""
import torch


ACTIVATIONS = {
    'relu': torch.nn.ReLU(inplace=True),
    'leaky_relu': torch.nn.LeakyReLU(inplace=True),
    'selu': torch.nn.SELU(),
}
