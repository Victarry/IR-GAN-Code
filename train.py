# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Training Loop script"""
import os
import glob

import torch
from torch.utils.data import DataLoader

from irgan.data.datasets import DATASETS
from irgan.evaluation.evaluate import Evaluator
from irgan.utils.config import keys, parse_config
from irgan.utils.visualize import VisdomPlotter, TensorboardPlotter
from irgan.models.models import MODELS
from irgan.data import codraw_dataset
from irgan.data import clevr_dataset
import time


class Trainer():
    def __init__(self, cfg):
        # the path to save generated images during training, default in logs/exp_name
        img_path = os.path.join(cfg.log_path, cfg.exp_name, 'train_images_*')
        if glob.glob(img_path):
            raise Exception('all directories with name train_images_* under '
                            'the experiment directory need to be removed')
        path = os.path.join(cfg.log_path, cfg.exp_name)

        self.model = MODELS[cfg.gan_type](cfg)
        if cfg.load_snapshot is not None:
            print('-----load pretrained model------')
            self.model.load_model(cfg.load_snapshot)

        self.model.save_model(path, epoch=0, iteration=0)

        self.dataset = DATASETS[cfg.dataset](path=keys[cfg.dataset],
                                             cfg=cfg,
                                             img_size=cfg.img_size)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=cfg.batch_size,
                                     shuffle=False,
                                     num_workers=cfg.num_workers,
                                     pin_memory=True,
                                     drop_last=True)

        if cfg.dataset == 'codraw':
            self.dataloader.collate_fn = codraw_dataset.collate_data
        elif cfg.dataset == 'iclevr':
            self.dataloader.collate_fn = clevr_dataset.collate_data

        self.visualizer = TensorboardPlotter(env_name=cfg.exp_name,
                                             server=cfg.vis_server)
        self.logger = None

        self.cfg = cfg

    def train(self):
        iteration_counter = 0
        for epoch in range(self.cfg.epochs):
            if cfg.dataset == 'codraw':
                self.dataset.shuffle()
            epoch_start_time = time.time()
            for batch in self.dataloader:
                if iteration_counter >= 0 and iteration_counter % self.cfg.save_rate == 0:
                    torch.cuda.empty_cache()
                    evaluator = Evaluator.factory(self.cfg, self.visualizer,
                                                  self.logger)
                    evaluator.evaluate(iteration_counter)
                    del evaluator

                iteration_counter += 1

                self.model.train_batch(batch, epoch, iteration_counter,
                                       self.visualizer, self.logger)
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, self.cfg.epochs, time.time() - epoch_start_time))


if __name__ == '__main__':
    cfg = parse_config()
    trainer = Trainer(cfg)
    trainer.train()
