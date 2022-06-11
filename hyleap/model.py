"""
Author: Dikshant Gupta
Time: 27.05.22 19:23
"""

import numpy as np
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        x = x.contiguous()
        return x.view(x.size(0), -1)


class ExperienceBuffer:
    def __init__(self, buffer_size=50):
        self.buffer = []
        self.buffer_size = buffer_size
        self.num_entries = 0

    def num_entries(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self):
        return self.buffer[np.random.randint(0, len(self.buffer))]


class HyLEAPNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # input_shape = (110, 310, 3)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=12*37*32, out_features=256),
            nn.ReLU()
        )
        self.rnn = nn.LSTMCell(input_size=256, hidden_size=128)
        self.action = nn.Linear(in_features=128, out_features=3)
        self.value = nn.Linear(in_features=128, out_features=1)

    def forward(self, x, h=None, c=None):
        x = self.net(x)
        if h is None:
            hx, cx = self.rnn(x)
        else:
            hx, cx = self.rnn(x, (h, c))
        act = self.action(hx)
        val = self.value(hx)

        return act, val, (hx, cx)
