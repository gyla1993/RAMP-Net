
import numpy as np
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, X):#B
        return self.linear(X)
    