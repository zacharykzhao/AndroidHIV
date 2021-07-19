# -*- coding: UTF-8 -*-

# author: Zachary Kaifa ZHAO
# e-mail: kaifa dot zhao (at) connet dot polyu dot hk
# datetime: 2021/7/12 4:12 PM
# software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F


class Substitute(nn.Module):
    def __init__(self, input_dimension):
        super(Substitute, self).__init__()
        self.fc1 = nn.Linear(input_dimension, 128)
        self.drop_out1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 128)
        self.drop_out2 = nn.Dropout(0.5)
        self.output = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop_out1(x)
        x = F.relu(self.fc2(x))
        x = self.drop_out2(x)
        x = F.softmax(self.output(x), dim=1)
        return x
# AdaGrad
