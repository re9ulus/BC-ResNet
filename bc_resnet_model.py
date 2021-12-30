import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torchaudio import transforms

import get_data
# import subspectram_norm


class MagicBlock(nn.Module):
    def __init__(self, n_input_chan=1, n_output_chan=1):
        super().__init__()
        self.n_input_chan = n_input_chan
        self.n_output_chan = n_output_chan
        self.scale_conv = nn.Conv2d(n_input_chan, n_output_chan, kernel_size=(1, 1))
        self.scale_bn = nn.BatchNorm2d(n_output_chan)
        
        self.vertical_conv1 = nn.Conv2d(n_output_chan, n_output_chan, kernel_size=(3, 1))
        self.vertical_conv2 = nn.Conv2d(n_output_chan, n_output_chan, kernel_size=(3, 1))
        self.horizontal_conv1 = nn.Conv1d(n_output_chan, n_output_chan, kernel_size=3, padding="same")
        self.horizontal_conv2 = nn.Conv1d(n_output_chan, n_output_chan, kernel_size=3, padding="same")

        self.bn = nn.BatchNorm1d(n_output_chan)  # TODO: Use subspectralnorm
        self.swish = nn.SiLU()
        self.one_x_one_conv = nn.Conv1d(n_output_chan, n_output_chan, kernel_size=1)

    def forward(self, x):
        if self.n_input_chan != self.n_output_chan:
            x = self.scale_conv(x)
            x = self.scale_bn(x)
            x = F.relu(x)

        n_freq = x.shape[2]

        y = self.vertical_conv1(x)
        y = self.vertical_conv2(y)
        y = torch.mean(y, dim=2)
        y = self.horizontal_conv1(y)
        y = self.horizontal_conv2(y)

        y = self.bn(y)
        y = self.swish(y)
        y = self.one_x_one_conv(y)
        
        y = y.unsqueeze(2).repeat(1, 1, n_freq, 1)

        return x + y


class BcResNetModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_conv = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(2, 1))

        self.m1 = MagicBlock(6, 3)
        self.m2 = MagicBlock(3, 3)
        self.m2 = MagicBlock(3, 1)
        
        self.head = nn.Sequential(
            nn.Linear(504, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 35)
        )
    
    def forward(self, x):
        x = self.input_conv(x)

        x = self.m1(x)
        x = F.relu(x)
        x = self.m2(x)
        x = nn.Flatten()(x)
        # print("flatten shape: ", x.shape)
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
