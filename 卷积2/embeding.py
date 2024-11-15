
import torch
from torch.nn import functional as F

embs = torch.randn(5,2)
print(embs)

tokens = torch.tensor([1,2,1,0,0])
print(embs[tokens])

a = F.one_hot(tokens,5).float()
print(a)
print(a@embs)

# embs = torch.nn.Embedding(10000,100)
# print(embs.weight.data)
# print(embs(tokens))
