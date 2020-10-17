# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
An inference time implementation for
recurrent GAN. Main difference is connecting
last time step generations to the next time
step. (Getting rid of Teacher Forcing)
"""
import os
from glob import glob

import torch
import torch.nn as nn
from torch.nn import DataParallel
import cv2

from ...models.networks.generator_factory import GeneratorFactory
from ...models.image_encoder import ImageEncoder
from ...models.sentence_encoder import SentenceEncoder
from ...models.condition_encoder import ConditionEncoder


class InferenceRecurrentGAN():
    def __init__(self, cfg):
        """A recurrent GAN model, each time step an generated image
        (x'_{t-1}) and the current question q_{t} are fed to the RNN
        to produce the conditioning vector for the GAN.
        The following equations describe this model:

            - c_{t} = RNN(h_{t-1}, q_{t}, x^{~}_{t-1})
            - x^{~}_{t} = G(z | c_{t})
        """
        super(InferenceRecurrentGAN, self).__init__()
        self.generator = DataParallel(GeneratorFactory.create_instance(cfg),
                                      device_ids=[0]).cuda()
        self.use_history = cfg.use_history
        # if use rnn to encode history instructions information for current step manipulation
        if self.use_history:
            self.rnn = nn.DataParallel(nn.GRU(cfg.input_dim,
                                              cfg.hidden_dim,
                                              batch_first=False),
                                       dim=1,
                                       device_ids=[0]).cuda()

        self.layer_norm = nn.DataParallel(nn.LayerNorm(cfg.hidden_dim),
                                          device_ids=[0]).cuda()
        self.use_image_encoder = cfg.use_fg
        if self.use_image_encoder:
            self.image_encoder = DataParallel(ImageEncoder(cfg),
                                              device_ids=[0]).cuda()

        self.condition_encoder = DataParallel(ConditionEncoder(cfg),
                                              device_ids=[0]).cuda()

        self.sentence_encoder = nn.DataParallel(SentenceEncoder(cfg),
                                                device_ids=[0]).cuda()

        self.cfg = cfg
        self.results_path = cfg.results_path
        if not os.path.exists(cfg.results_path):
            os.mkdir(cfg.results_path)

    def predict(self, batch):
        with torch.no_grad():
            batch_size = len(batch['image'])
            max_seq_len = batch['image'].size(1)
            scene_id = batch['scene_id']
            dialog_lengths = batch['dialog_length']
            # Initial inputs for the RNN set to zeros
            prev_image = torch.FloatTensor(batch['background']).unsqueeze(0) \
                .repeat(batch_size, 1, 1, 1)
            hidden = torch.zeros(1, batch_size, self.cfg.hidden_dim)
            generated_images = []
            # save all generated image tensors in each time step into generated_images
            for t in range(max_seq_len):
                turns_word_embedding = batch['turn_word_embedding'][:, t]
                turns_lengths = batch['turn_lengths'][:, t]
                image_vec = None
                image_feature_map = None
                if self.use_image_encoder:
                    image_feature_map, image_vec = self.image_encoder(
                        prev_image)

                turn_embedding, _ = self.sentence_encoder(
                    turns_word_embedding, turns_lengths)
                rnn_condition = self.condition_encoder(turn_embedding,
                                                       image_vec)
                if self.use_history:
                    rnn_condition = rnn_condition.unsqueeze(0)
                    output, hidden = self.rnn(rnn_condition, hidden)
                else:
                    output = rnn_condition
                output = output.squeeze(0)
                output = self.layer_norm(output)

                generated_image = self._forward_generator(
                    batch_size, output, image_feature_map)
                generated_images.append(generated_image)
                prev_image = generated_image
        # save images tensors into file, whether save all images or images at last timestep can be determined by config, the reported results in our paper is only for images at last timestep.
        _save_predictions(generated_images, batch['turn'], scene_id,
                          self.results_path, batch['image'], dialog_lengths,
                          self.cfg.inference_save_last_only)

    def _forward_generator(self, batch_size, condition, image_feature_maps):
        # In our experiments, the noise make little difference for the model performance, so we set it to zeros.
        noise = torch.FloatTensor(batch_size,
                                  self.cfg.noise_dim).zero_().cuda()

        fake_images, _, _, _ = self.generator(noise, condition,
                                              image_feature_maps)

        return fake_images

    def load(self, pre_trained_path, iteration=None):
        snapshot = _read_weights(pre_trained_path, iteration)

        self.generator.load_state_dict(snapshot['generator_state_dict'])
        if self.use_history:
            self.rnn.load_state_dict(snapshot['rnn_state_dict'])
        self.layer_norm.load_state_dict(snapshot['layer_norm_state_dict'])
        if self.use_image_encoder:
            self.image_encoder.load_state_dict(
                snapshot['image_encoder_state_dict'])
        self.condition_encoder.load_state_dict(
            snapshot['condition_encoder_state_dict'])
        self.sentence_encoder.load_state_dict(
            snapshot['sentence_encoder_state_dict'])


def _save_predictions(images, text, scene_id, results_path, gt_images,
                      dialog_lengths, save_last_only):
    """
    Be careful to the case where dialog sequences with different length in a single batch.

    @args:
        images:  (seq_len, batch_size) generated images at different step.
        text:    (batch_size, seq_len)
        scene_id:(batch_size,) id of each sequence
        results_path: string, the name of saved path
        gt_images: (batch_size, seq_len) ground truth images
        dialog_lengths: (batch_size, ) length of each sequence
    """
    max_seq_len = len(images)
    # Each time, save images for one dialog
    for i, scene in enumerate(scene_id):
        if not os.path.exists(os.path.join(results_path, str(scene))):
            os.mkdir(os.path.join(results_path, str(scene)))
        if not os.path.exists(os.path.join(results_path, str(scene) + '_gt')):
            os.mkdir(os.path.join(results_path, str(scene) + '_gt'))
        if save_last_only:
            t = dialog_lengths[i] - 1
        else:
            t = 0
        while t < dialog_lengths[i]:
            image = (images[t][i].data.cpu().numpy() + 1) * 128
            image = image.transpose(1, 2, 0)[..., ::-1]
            # linux limit file name length up to 256, so we need truncate the file name.
            # While to be compared with GeNeVA, we leave the source code unchanged.
            query = text[i][t]
            # query = text[i][t][:200]
            gt_image = (gt_images[i][t].data.cpu().numpy() + 1) * 128
            # opencv needs images channel arranged as (H, W, C) and 'BGR'
            gt_image = gt_image.transpose(1, 2, 0)[..., ::-1]
            cv2.imwrite(
                os.path.join(results_path, str(scene),
                             '{}_{}.png'.format(t, query)), image)
            cv2.imwrite(
                os.path.join(results_path,
                             str(scene) + '_gt', '{}_{}.png'.format(t, query)),
                gt_image)
            t += 1


def _read_weights(pre_trained_path, iteration):
    if iteration is None:
        iteration = ''
    iteration = str(iteration)
    try:
        snapshot = torch.load(
            glob('{}/snapshot_{}*'.format(pre_trained_path, iteration))[0])
    except IndexError:
        snapshot = torch.load(pre_trained_path)
    return snapshot
