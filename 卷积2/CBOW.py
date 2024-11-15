import torch

from torch import nn

class CBOW(nn.Module):

    def __init__(self):
        super().__init__()

        # self.embs = nn.Parameter(torch.randn(10000,128))
        self.embs = torch.nn.Embedding(10000,128)

        self.layer = nn.Sequential(
            nn.Linear(4*128,128,bias=False)
        )

        self.loss_fn = nn.MSELoss()

    def forwad(self,x1,x2,y):
        _x1 = self.embs(x1)
        _x2 = self.embs(x2)

        _x = torch.concat((_x1,_x2),dim=1).reshape(-1,4*128)

        _y = self.layer(_x)

        return self.loss_fn(_y,y)
    

#train......

# net = CBOW()

# x1,x2,y = None

# loss = net(x1,x2,y)

# opt.grad_zero()
# loss.backward()
# opt.step()

