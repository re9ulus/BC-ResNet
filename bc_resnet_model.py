import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torchaudio import transforms

import get_data
# import subspectram_norm

# TODO: Add dilation to conv

DROPOUT = 0.4


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
            nn.Dropout2d(DROPOUT)
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

        if stride == 1:
            conv = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 1), groups=out_chan, padding="same")
        else:
            conv = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 1), stride=(stride, 1), groups=out_chan, padding=(1, 0))

        self.f2 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            conv,
            # TODO: use subspectral norm instead
            nn.BatchNorm2d(out_chan),
        )

        self.f1 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=(1, 3), padding="same", groups=out_chan),
            nn.BatchNorm2d(out_chan),
            nn.SiLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=1),
            nn.Dropout2d(DROPOUT)
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
    def __init__(self, n_class=35):
        super().__init__()

        self.input_conv = nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 1), padding=2)

        self.t1 = TransitionBlock(16, 8)
        self.n11 = NormalBlock(8)

        self.t2 = TransitionBlock(8, 12, stride=2)  # dilation (1, 2)
        self.n21 = NormalBlock(12) 

        self.t3 = TransitionBlock(12, 16, stride=2)
        self.n31 = NormalBlock(16)
        self.n32 = NormalBlock(16)
        self.n33 = NormalBlock(16)

        self.t4 = TransitionBlock(16, 20)  # dilation (1, 8)
        self.n41 = NormalBlock(20)
        self.n42 = NormalBlock(20)
        self.n43 = NormalBlock(20)

        self.dw_conv = nn.Conv2d(20, 20, kernel_size=(5, 5), dilation=1, groups=20)
        self.onexone_conv = nn.Conv2d(20, 32, kernel_size=1)

        self.head_conv = nn.Conv2d(32, n_class, kernel_size=1)
    
    def forward(self, x):
        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)

        x = self.t2(x)
        x = self.n21(x)

        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)

        x = torch.mean(x, dim=3, keepdim=True)
        x = self.head_conv(x)

        x = x.squeeze()

        return F.log_softmax(x, dim=-1)


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
        resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)

        tensors += [(to_mel(resample(waveform)) + eps).log2()]
        targets += [get_data.label_to_idx(label)]

    tensors = pad_sequence(tensors)
    targets = torch.LongTensor(targets)

    return tensors, targets
