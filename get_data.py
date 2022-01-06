import os
import torch
from torchaudio import transforms
from torchaudio.datasets import SPEECHCOMMANDS


EPS = 1e-9
SAMPLE_RATE = 16000
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


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str, path="./"):
        super().__init__(path, download=True)
        self.to_mel = transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, f_max=8000, n_mels=40)

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
        if sample_rate != SAMPLE_RATE: 
            resampler = transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
        log_mel = (self.to_mel(waveform) + EPS).log2()

        return log_mel, label


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
    tensors, targets = [], []
    for log_mel, label in batch:
        tensors.append(log_mel)
        targets.append(label_to_idx(label))

    tensors = pad_sequence(tensors)
    targets = torch.LongTensor(targets)

    return tensors, targets
