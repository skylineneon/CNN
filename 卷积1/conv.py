import torch
from torch import nn

img = torch.randn(1,6,32,32)

conv = nn.Conv2d(6,3,3,1,1,groups=3)

y = conv(img)
print(y.shape)