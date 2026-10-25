# coding: utf-8
"""Masked-phoneme / word-prediction dataset.

Each record is ``{'phonemes': [per-word phoneme strings],
'input_ids': [per-word vocabulary ids]}``, already phonemized on the Linux
box. Nothing here needs eSpeak or the phonemizer package.

Per sample the loader:
  1. expands every word into a character stream, inserting a separator space,
  2. masks whole words with probability ``word_mask_prob``,
  3. random-crops ``max_mel_length`` characters,
  4. returns the masked stream, the clean stream, and the per-character word id.
"""

import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from text_utils import TextCleaner

np.random.seed(1)


class PhonemeDataset(torch.utils.data.Dataset):
    def __init__(self, records, token_maps, word_separator,
                 token_separator=" ", token_mask="M", max_mel_length=512,
                 word_mask_prob=0.15, phoneme_mask_prob=0.1,
                 replace_prob=0.2):
        self.data = records
        self.max_mel_length = max_mel_length
        self.word_mask_prob = word_mask_prob
        self.phoneme_mask_prob = phoneme_mask_prob
        self.replace_prob = replace_prob
        self.text_cleaner = TextCleaner()
        self.word_separator = word_separator
        self.token_separator = token_separator
        self.token_mask = token_mask
        if isinstance(token_maps, str):
            with open(token_maps, 'rb') as handle:
                token_maps = pickle.load(handle)
        self.token_maps = token_maps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        phonemes = self.data[idx]['phonemes']
        input_ids = self.data[idx]['input_ids']

        words, labels, phoneme = [], "", ""
        phoneme_list = ''.join(phonemes)
        masked_index = []

        for word_phonemes, word_id in zip(phonemes, input_ids):
            words.extend([word_id] * len(word_phonemes))
            words.append(self.word_separator)
            labels += word_phonemes + self.token_separator

            if np.random.rand() < self.word_mask_prob:
                if np.random.rand() < self.replace_prob:
                    if np.random.rand() < (self.phoneme_mask_prob
                                           / self.replace_prob):
                        # Replace with random phonemes drawn from this record.
                        phoneme += ''.join(
                            phoneme_list[np.random.randint(0, len(phoneme_list))]
                            for _ in range(len(word_phonemes))
                        )
                    else:
                        phoneme += word_phonemes          # left intact
                else:
                    phoneme += self.token_mask * len(word_phonemes)
                masked_index.extend(
                    range(len(phoneme) - len(word_phonemes), len(phoneme))
                )
            else:
                phoneme += word_phonemes
            phoneme += self.token_separator

        mel_length = len(phoneme)
        masked_idx = np.array(masked_index, dtype=np.int64)
        masked_index = []
        if mel_length > self.max_mel_length:
            start = np.random.randint(0, mel_length - self.max_mel_length)
            stop = start + self.max_mel_length
            phoneme = phoneme[start:stop]
            words = words[start:stop]
            labels = labels[start:stop]
            masked_index = [
                int(m - start) for m in masked_idx if start <= m < stop
            ]
        else:
            masked_index = masked_idx.tolist()

        phoneme = self.text_cleaner(phoneme)
        labels = self.text_cleaner(labels)
        words = [self.token_maps[w]['token'] for w in words]

        assert len(phoneme) == len(words) == len(labels), (
            "stream lengths diverged -- a phoneme string was probably empty"
        )

        return (
            torch.LongTensor(phoneme),
            torch.LongTensor(words),
            torch.LongTensor(labels),
            masked_index,
        )


class Collater:
    def __call__(self, batch):
        batch_size = len(batch)
        lengths = [b[0].shape[0] for b in batch]
        order = np.argsort(lengths)[::-1]
        batch = [batch[i] for i in order]

        max_len = max(b[0].shape[0] for b in batch)
        phonemes = torch.zeros((batch_size, max_len)).long()
        words = torch.zeros((batch_size, max_len)).long()
        labels = torch.zeros((batch_size, max_len)).long()
        input_lengths, masked_indices = [], []

        for bid, (phoneme, word, label, masked_index) in enumerate(batch):
            size = phoneme.size(0)
            phonemes[bid, :size] = phoneme
            words[bid, :size] = word
            labels[bid, :size] = label
            input_lengths.append(size)
            masked_indices.append(masked_index)

        return words, labels, phonemes, input_lengths, masked_indices


def build_dataloader(records, token_maps, word_separator, batch_size=8,
                     num_workers=0, shuffle=True, pin_memory=True,
                     dataset_config=None):
    dataset = PhonemeDataset(
        records, token_maps, word_separator, **(dataset_config or {})
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,
        collate_fn=Collater(),
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers),
    )
