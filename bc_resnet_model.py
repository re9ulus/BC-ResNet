import torch
from torch import nn
import torch.nn.functional as F


DROPOUT = 0.1


class NormalBlock(nn.Module):
    def __init__(self, n_chan, *, dilation=1, dropout=DROPOUT):
        super().__init__()
        self.f2 = nn.Sequential(
            nn.Conv2d(n_chan, n_chan, kernel_size=(3, 1), padding="same", groups=n_chan),
            # TODO: use subspectral norm instead
            nn.BatchNorm2d(n_chan),
        )
        self.f1 = nn.Sequential(
            nn.Conv2d(n_chan, n_chan, kernel_size=(1, 3), padding="same", groups=n_chan, dilation=(1, dilation)),
            nn.BatchNorm2d(n_chan),
            nn.SiLU(),
            nn.Conv2d(n_chan, n_chan, kernel_size=1),
            nn.Dropout2d(dropout)
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
    def __init__(self, in_chan, out_chan, *, dilation=1, stride=1, dropout=DROPOUT):
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
            nn.Conv2d(out_chan, out_chan, kernel_size=(1, 3), padding="same", groups=out_chan, dilation=(1, dilation)),
            nn.BatchNorm2d(out_chan),
            nn.SiLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=1),
            nn.Dropout2d(dropout)
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

        self.t2 = TransitionBlock(8, 12, dilation=2, stride=2)
        self.n21 = NormalBlock(12, dilation=2) 

        self.t3 = TransitionBlock(12, 16, dilation=4, stride=2)
        self.n31 = NormalBlock(16, dilation=4)
        self.n32 = NormalBlock(16, dilation=4)
        self.n33 = NormalBlock(16, dilation=4)

        self.t4 = TransitionBlock(16, 20, dilation=8)
        self.n41 = NormalBlock(20, dilation=8)
        self.n42 = NormalBlock(20, dilation=8)
        self.n43 = NormalBlock(20, dilation=8)

        self.dw_conv = nn.Conv2d(20, 20, kernel_size=(5, 5), groups=20)
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
