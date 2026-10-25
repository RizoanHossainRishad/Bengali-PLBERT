# coding: utf-8
"""PL-BERT multi-task model.

Identical in structure to yl4579/PL-BERT's model.py. Keeping the attribute
name ``encoder`` matters: StyleTTS2's ``load_plbert()`` selects checkpoint keys
by the ``encoder.`` prefix and throws everything else away.
"""

import torch.nn as nn


class MultiTaskModel(nn.Module):
    def __init__(self, model, num_tokens=178, num_vocab=50002, hidden_size=768):
        super().__init__()
        self.encoder = model                                   # AlbertModel
        self.mask_predictor = nn.Linear(hidden_size, num_tokens)
        self.word_predictor = nn.Linear(hidden_size, num_vocab)

    def forward(self, phonemes, attention_mask=None):
        output = self.encoder(phonemes, attention_mask=attention_mask)
        tokens_pred = self.mask_predictor(output.last_hidden_state)
        words_pred = self.word_predictor(output.last_hidden_state)
        return tokens_pred, words_pred
