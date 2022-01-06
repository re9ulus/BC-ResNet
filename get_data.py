import os
import numpy as np
import torch
from torchaudio import transforms
from torchaudio.datasets import SPEECHCOMMANDS


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, path="./", subset: str = None):
        super().__init__(path, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fh:
                return [
                    os.path.join(self._path, line.strip()) for line in fh
                ]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

    def __getitem__(self, n):
        waveform, sample_rate, label, _, _ = super().__getitem__(n)
        return waveform, sample_rate, label

    
    
LABELS = [
    'backward',
    'bed',
    'bird',
    'cat',
    'dog',
    'down',
    'eight',
    'five',
    'follow',
    'forward',
    'four',
    'go',
    'happy',
    'house',
    'learn',
    'left',
    'marvin',
    'nine',
    'no',
    'off',
    'on',
    'one',
    'right',
    'seven',
    'sheila',
    'six',
    'stop',
    'three',
    'tree',
    'two',
    'up',
    'visual',
    'wow',
    'yes',
    'zero'
]


_label_to_idx = {label: i for i, label in enumerate(LABELS)}
_idx_to_label = {i: label for label, i in _label_to_idx.items()}
    

def label_to_idx(label):
    return _label_to_idx[label]


def idx_to_label(idx):
    return _idx_to_label[idx]


def pad_sequence(batch):
    batch = [item.permute(2, 1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return batch.permute(0, 3, 2, 1) 


def collate_fn(batch):
    new_sample_rate = 16000
    to_mel = transforms.MelSpectrogram(sample_rate=new_sample_rate, n_fft=1024, f_max=8000, n_mels=40)
    tensors, targets = [], []
    eps = 1e-9
    for waveform, sample_rate, label in batch:
        resample = transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)

        tensors += [(to_mel(resample(waveform)) + eps).log2()]
        targets += [label_to_idx(label)]

    tensors = pad_sequence(tensors)
    targets = torch.LongTensor(targets)

    return tensors, targets


# Test shift augmentation
# where to put ??
def collate_fn_train(batch):
    new_sample_rate = 16000
    to_mel = transforms.MelSpectrogram(sample_rate=new_sample_rate, n_fft=1024, f_max=8000, n_mels=40)
    tensors, targets = [], []
    eps = 1e-9
    for waveform, sample_rate, label in batch:
        resample = transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)

        shift = np.random.randint(-1600, 1600)
        waveform = torch.roll(waveform, shift)
        if shift > 0:
            waveform[:shift] = 0
        elif shift < 0:
            waveform[-shift:] = 0

        tensors += [(to_mel(resample(waveform)) + eps).log2()]
        targets += [label_to_idx(label)]

    tensors = pad_sequence(tensors)
    targets = torch.LongTensor(targets)

    return tensors, targets
