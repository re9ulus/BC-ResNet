import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torchaudio import transforms

import get_data
# import subspectram_norm

# TODO: Add dilation to conv


class NormalBlock(nn.Module):
    def __init__(self, n_chan):
        super().__init__()
        self.f2 = nn.Sequential(
            nn.Conv2d(n_chan, n_chan, kernel_size=(3, 1), padding="same", groups=n_chan),
            # TODO: use subspectral norm instead
            nn.BatchNorm2d(n_chan),
        )
        self.f1 = nn.Sequential(
            nn.Conv2d(n_chan, n_chan, kernel_size=(1, 3), padding="same", groups=n_chan),
            nn.BatchNorm2d(n_chan),
            nn.SiLU(),
            nn.Conv2d(n_chan, n_chan, kernel_size=1),
            # TODO: Add dropout
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        n_freq = x.shape[2]
        x1 = self.f2(x)

        x2 = torch.mean(x1, dim=2, keepdim=True)
        x2 = self.f1(x2)
        x2 = x2.repeat(1, 1, n_freq, 1)

        return self.activation(x + x1 + x2)


class TransitionBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super().__init__()
        self.f2 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=(3, 1), stride=(stride, 1), groups=out_chan),
            # TODO: use subspectral norm instead
            nn.BatchNorm2d(out_chan),
        )

        self.f1 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=(1, 3), padding="same", groups=out_chan),
            nn.BatchNorm2d(out_chan),
            nn.SiLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=1),
            # TODO: Add dropout
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.f2(x)
        n_freq = x.shape[2]
        x1 = torch.mean(x, dim=2, keepdim=True)
        x1 = self.f1(x1)
        x1 = x1.repeat(1, 1, n_freq, 1)

        return self.activation(x + x1)


class BcResNetModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_conv = nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 1), dilation=1)

        self.m1 = TransitionBlock(16, 8)  # dilation 1
        self.m2 = NormalBlock(8)          # dilation 1

        self.m3 = TransitionBlock(8, 12, stride=2)  # dilation (1, 2)
        self.m4 = NormalBlock(12) 

        self.m5 = TransitionBlock(12, 16, stride=2)
        self.m6 = NormalBlock(16)
        self.m7 = NormalBlock(16)
        self.m8 = NormalBlock(16)

        self.m9 = TransitionBlock(16, 20)  # dilation (1, 8)
        self.m10 = NormalBlock(20)
        self.m11 = NormalBlock(20)
        self.m12 = NormalBlock(20)

        self.dw_conv = nn.Conv2d(20, 20, kernel_size=(5, 5), dilation=1, groups=20)
        self.onexone_conv = nn.Conv2d(20, 32, kernel_size=1)

        # self.m2 = TransitionBlock(3, 3, stride=2)
        # self.m3 = NormalBlock(3)
        # self.m4 = TransitionBlock(3, 1)
        
        self.head = nn.Sequential(
            nn.Linear(896, 64),
            nn.ReLU(),
            nn.Linear(64, 35)
        )
    
    def forward(self, x):
        x = self.input_conv(x)
        x = self.m1(x)
        x = self.m2(x)
        x = self.m3(x)
        x = self.m4(x)
        x = self.m5(x)
        x = self.m6(x)
        x = self.m7(x)
        x = self.m8(x)
        x = self.m9(x)
        x = self.m10(x)
        x = self.m11(x)
        x = self.m12(x)

        # x = self.dw_conv(x)
        x = self.onexone_conv(x)

        x = nn.Flatten()(x)
        x = self.head(x)

        return F.log_softmax(x, dim=-1)


def pad_sequence(batch):
    batch = [item.permute(2, 1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return batch.permute(0, 3, 2, 1) 


def collate_fn(batch):
    new_sample_rate = 16000
    # TODO: Use log mel_spec
    to_mel = transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, f_max=8000, n_mels=40)
    tensors, targets = [], []
    for waveform, sample_rate, label in batch:
        resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)

        tensors += [to_mel(resample(waveform))]
        targets += [get_data.label_to_idx(label)]

    tensors = pad_sequence(tensors)
    targets = torch.LongTensor(targets)

    return tensors, targets
