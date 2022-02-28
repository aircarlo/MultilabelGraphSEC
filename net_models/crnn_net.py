import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_chan, out_chan, ker_size, stride, pool_size):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, ker_size, stride, padding=ker_size[0] // 2, bias=True)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.mpool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        out = self.mpool(x)
        return out


class CRNN(nn.Module):
    def __init__(self, out_dim=200):  # default out_dim = 200 (num classes)
        super(CRNN, self).__init__()
        self.conv1 = conv_block(1, 128, ker_size=(5, 5), stride=1, pool_size=(5, 2))
        self.conv2 = conv_block(128, 128, ker_size=(5, 5), stride=1, pool_size=(4, 2))
        self.conv3 = conv_block(128, 128, ker_size=(5, 5), stride=1, pool_size=(3, 2))
        self.rnn1 = nn.GRU(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        # output dense
        self.dense = nn.Linear(128, out_dim)

    def forward(self, x, graph_in):
        # input is freq-time patches of size 96x101
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)       # [B,C,H,W] with h=1
        x = x.squeeze(2)        # [B,C,W]
        x = x.permute(0, 2, 1)  # [B,W,C]  # batch first = True
        x, _ = self.rnn1(x)
        # output dense
        x = self.dense(x[:,-1,:])
        return x