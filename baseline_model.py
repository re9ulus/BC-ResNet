import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

import get_data


class M5(nn.Module):
    def __init__(self, n_input=1, n_outputs=35, stride=16, n_channels=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channels, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channels)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channels, n_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channels)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channels, 2*n_channels, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channels)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channels, 2 * n_channels, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channels)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channels, n_outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    new_sample_rate = 8000
    tensors, targets = [], []
    for waveform, sample_rate, label in batch:
        sample_rate  # TODO: Use in transform, for resampling

        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)

        tensors += [transform(waveform)]
        targets += [get_data.label_to_idx(label)]

    tensors = pad_sequence(tensors)
    targets = torch.LongTensor(targets)  # torch.stack

    return tensors, targets


# placeholder

