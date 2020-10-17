# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""A sentence encoder that trains a GRU on top a sequence
of Glove word embedding"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentenceEncoder(nn.Module):
    """A sentence encoder. Takes input list of Glove word
    embedding, a GRU is trained on top, the final hidden state
    is used as the sentence encoding."""
    def __init__(self, cfg):
        super(SentenceEncoder, self).__init__()
        self.gru = nn.GRU(300,
                          cfg.embedding_dim // 2,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)

        self.layer_norm = nn.LayerNorm(cfg.embedding_dim)
        self.cfg = cfg

    def forward(self, words, lengths):
        """
        words:   (batch_size, seq_len, embed_dim)
        lengths: (batch_size, )
        """
        total_length = words.size(1)
        lengths = lengths.long()
        reorder = False
        sorted_len, indices = torch.sort(lengths, descending=True)
        if not torch.equal(sorted_len, lengths):
            _, reverse_sorting = torch.sort(indices)
            reorder = True
            words = words[indices]
            lengths = lengths[indices]

        lengths[lengths == 0] = 1

        packed_padded_sequence = pack_padded_sequence(words,
                                                      lengths,
                                                      batch_first=True)

        self.gru.flatten_parameters()
        embeddings, h = self.gru(packed_padded_sequence)
        h = h.permute(1, 0, 2).contiguous()
        h = h.view(-1, self.cfg.embedding_dim)
        embeddings, lengths = pad_packed_sequence(embeddings,
                                                  batch_first=True,
                                                  total_length=total_length)
        if reorder:
            h = h[reverse_sorting]

        h = self.layer_norm(h)
        return h, embeddings
