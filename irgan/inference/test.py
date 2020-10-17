# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Testing loop script"""
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..data.datasets import DATASETS
from ..evaluation.evaluate_metrics import report_inception_objects_score
from ..utils.config import keys, parse_config
from ..models.models import INFERENCE_MODELS
from ..data import codraw_dataset
from ..data import clevr_dataset


class Tester():
    def __init__(self, cfg, use_val=False, iteration=None, test_eval=False):
        self.model = INFERENCE_MODELS[cfg.gan_type](cfg)

        if use_val:
            dataset_path = cfg.val_dataset
            model_path = os.path.join(cfg.log_path, cfg.exp_name)
        else:
            dataset_path = cfg.dataset
            model_path = cfg.load_snapshot
        if test_eval:
            dataset_path = cfg.test_dataset
            model_path = cfg.load_snapshot
        self.model.load(model_path, iteration)
        print('use dataset %s, use model' % dataset_path, model_path)
        self.dataset = DATASETS[cfg.dataset](path=keys[dataset_path],
                                             cfg=cfg,
                                             img_size=cfg.img_size)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=cfg.batch_size,
                                     shuffle=False,
                                     num_workers=cfg.num_workers,
                                     drop_last=True)

        self.iterations = len(self.dataset) // cfg.batch_size

        if cfg.dataset == 'codraw':
            self.dataloader.collate_fn = codraw_dataset.collate_data
        elif cfg.dataset == 'iclevr':
            self.dataloader.collate_fn = clevr_dataset.collate_data

        if cfg.results_path is None:
            cfg.results_path = os.path.join(cfg.log_path, cfg.exp_name,
                                            'results')
            if not os.path.exists(cfg.results_path):
                os.mkdir(cfg.results_path)

        self.cfg = cfg
        self.dataset_path = dataset_path

    def test(self):
        for batch in tqdm(self.dataloader, total=self.iterations):
            self.model.predict(batch)
