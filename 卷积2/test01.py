import torch


a = torch.tensor([-1,2,3]).cuad()

a = torch.relu(a)

print(a)

