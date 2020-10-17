# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Models Manager"""
from irgan.models.recurrent_gan import RecurrentGAN
from irgan.models.inference_models.recurrent_gan import InferenceRecurrentGAN
MODELS = {
    'recurrent_gan': RecurrentGAN
}


INFERENCE_MODELS = {
    'recurrent_gan': InferenceRecurrentGAN
}
