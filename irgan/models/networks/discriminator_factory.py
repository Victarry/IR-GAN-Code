# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Discriminator Factory for creating a discriminator instance
with specific properties.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from ...definitions.activations import ACTIVATIONS
from ...definitions.res_blocks import ResDownBlock


class DiscriminatorFactory():
    @staticmethod
    def create_instance(cfg):
        """Creates an instance of a discriminator that matches
        the given arguments.
        Args:
              cfg: Experiment configurations. This method needs
              [model, arch, conditioning] flags.
        Returns:
             discriminator: discriminator instance that implements nn.Module
        """
        if cfg.gan_type == 'recurrent_gan':
            return DiscriminatorAdditiveGANRes(cfg)
        elif cfg.gan_type == 'cycle_gan':
            return DiscriminatorAdditiveGANRes(cfg)
        else:
            raise Exception('no such model')


class DiscriminatorAdditiveGANRes(nn.Module):
    """
    Additive Discriminator for 128x128 image generation, conditioned
    on bag-of-words/caption by projection.
    """

    def __init__(self, cfg):
        super(DiscriminatorAdditiveGANRes, self).__init__()

        self.activation = ACTIVATIONS[cfg.activation]

        self.resdown1 = ResDownBlock(3, 64, downsample=True, first_block=True,
                                     activation=cfg.activation,
                                     use_spectral_norm=cfg.disc_sn)
        # state size. (64) x 64 x 64
        self.resdown2 = ResDownBlock(64, 128, downsample=True,
                                     activation=cfg.activation,
                                     use_spectral_norm=cfg.disc_sn)
        # state size. (128) x 32 x 32
        self.resdown3 = ResDownBlock(128, 256, downsample=True,
                                     activation=cfg.activation,
                                     use_spectral_norm=cfg.disc_sn)

        extra_channels = 0
        if cfg.use_fd:
            if cfg.disc_img_conditioning == 'concat' or cfg.disc_img_conditioning == 'res':
                extra_channels += 256

        if cfg.conditioning == 'concat':
            extra_channels += cfg.disc_cond_channels

        # state size. (256) x 16 x 16
        self.resdown4 = ResDownBlock(256 + extra_channels, 512,
                                     downsample=True,
                                     self_attn=cfg.self_attention,
                                     activation=cfg.activation,
                                     use_spectral_norm=cfg.disc_sn)
        # state size. (512) x 8 x 8
        self.resdown5 = ResDownBlock(512, 1024,
                                     downsample=True,
                                     activation=cfg.activation,
                                     use_spectral_norm=cfg.disc_sn)

        # state size. (1024) x 4 x 4
        self.resdown6 = ResDownBlock(1024, 1024, downsample=False,
                                     activation=cfg.activation,
                                     use_spectral_norm=cfg.disc_sn)

        self.linear = torch.nn.Linear(1024, 1)
        if cfg.disc_sn:
            self.linear = spectral_norm(self.linear)

        condition_dim = cfg.hidden_dim

        if cfg.conditioning == 'projection':
            self.condition_projector = nn.Sequential(
                torch.nn.Linear(condition_dim, 1024),
                nn.ReLU(),
                torch.nn.Linear(1024, 1024),
            )
        elif cfg.conditioning == 'concat':
            self.condition_projector = nn.Sequential(
                torch.nn.Linear(condition_dim, 1024),
                nn.ReLU(),
                torch.nn.Linear(1024, cfg.disc_cond_channels),
            )
        if cfg.aux_reg > 0:
            self.aux_objective = nn.Sequential(
                torch.nn.Linear(1024, 256),
                nn.ReLU(),
                torch.nn.Linear(256, cfg.num_objects),
            )

        self.cfg = cfg

    def forward(self, x, y, prev_image):
        """
        x: current image
        y: condition vector output by rnn encoder
        prev_image: previous real image

        cfg args:
            cfg.conditioning:   ways to use instruction condition
                1. projection: cGANs with projection, use sum as output
                2. concat: like text2image, concat repeated condition at middle layer
            cdg.disc_image_conditioning: ways to use previous image feature
                1. concat
                2. subtract
                3. res
        """
        if self.cfg.use_fd:
            prev_image = self.resdown1(prev_image)
            prev_image = self.resdown2(prev_image)
            prev_image = self.resdown3(prev_image)

        y = self.condition_projector(y)

        x = self.resdown1(x)
        x = self.resdown2(x)
        x = self.resdown3(x)

        if self.cfg.use_fd:
            if self.cfg.disc_img_conditioning == 'concat':
                x = torch.cat((x, prev_image), dim=1)
            elif self.cfg.disc_img_conditioning == 'subtract':
                x = x - prev_image
            elif self.cfg.disc_img_conditioning == 'res':
                x = torch.cat((x, x - prev_image), dim=1)

        if self.cfg.conditioning == 'concat':
            H, W = x.size(2), x.size(3)
            y = y.repeat(H, W, 1, 1).permute(2, 3, 0, 1)
            x = torch.cat([x, y], dim=1)

        x = self.resdown4(x)
        x = self.resdown5(x)
        x = self.resdown6(x)

        intermediate_features = x

        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))  # Global Sum Pooling
        if self.cfg.aux_reg > 0:
            aux = F.sigmoid(self.aux_objective(x))
        else:
            aux = None

        out = self.linear(x).squeeze(1)

        if self.cfg.conditioning == 'projection':
            c = torch.sum(y * x, dim=1)
            out = out + c

        return out, aux, intermediate_features
