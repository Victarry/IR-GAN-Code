# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Datasets Manager"""
from . import clevr_dataset
from . import codraw_dataset


DATASETS = {
    'codraw': codraw_dataset.CoDrawDataset,
    'iclevr': clevr_dataset.ICLEVERDataset,
}
