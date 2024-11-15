import torch
from torch import nn

class MyMSE(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self,y,target):
        return torch.mean((y-target)**2)