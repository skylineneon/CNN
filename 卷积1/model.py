import torch
from torch.nn import init

class MnistNet(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        # self.f1 = torch.nn.Linear(28*28,64)
        # self.f1_act = torch.nn.LeakyReLU()
        # self.f2 = torch.nn.Linear(64,10)
        # self.f2_act = torch.nn.Softmax(-1)

        # self.fc = torch.nn.Sequential(
        #     torch.nn.Linear(28*28,512),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(512,10),
        #     torch.nn.Softmax(-1)
        # )

        # self._layer = torch.nn.Sequential(
        #     torch.nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=1,padding=1), #out:(N,3,28,28)
        #     torch.nn.SiLU(),
        #     torch.nn.MaxPool2d(2,2), #out:(N,3,14,14)
        #     torch.nn.BatchNorm2d(3),
        #     torch.nn.Conv2d(3,6,3,1,padding=1), #out:(N,6,14,14)
        #     torch.nn.SiLU(),
        #     torch.nn.MaxPool2d(2,2), #out:(N,3,7,7)
        #     torch.nn.BatchNorm2d(6),
        #     torch.nn.Conv2d(6,12,3,1,padding=1), #out:(N,12,7,7)
        #     torch.nn.SiLU(),
        #     torch.nn.MaxPool2d(2,2),
        #     torch.nn.Dropout2d(0.2),
        #     torch.nn.BatchNorm2d(12),
        #     torch.nn.Conv2d(12,24,3,1,padding=1), #out:(N,24,3,3)
        #     torch.nn.SiLU(),
        #     torch.nn.Flatten(),
        #     torch.nn.Dropout(0.2),
        #     torch.nn.BatchNorm1d(216),
        #     torch.nn.Linear(216,216),
        #     torch.nn.SiLU(),
        #     torch.nn.BatchNorm1d(216),
        #     torch.nn.Linear(216,10),
        # )

        self._input_layer = torch.nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=1,padding=1) #out:(N,3,28,28)

        self._layer_01 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3,3,3,1,1),
            torch.nn.SiLU()
        )

        self._max_pool = torch.nn.MaxPool2d(14,14)

        self._output_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(12),
            torch.nn.Linear(12,10,bias=False)
        )




        self.apply(self._init_weight)

    
    def _init_weight(self,m):
        if isinstance(m, torch.nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Linear):
            # init.kaiming_normal_(m.weight)
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)


        # self._layer = torch.nn.Sequential(
        #     torch.nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=2),
        #     torch.nn.SiLU(),
        #     torch.nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=2),
        #     torch.nn.SiLU(),
        #     torch.nn.Conv2d(in_channels=6,out_channels=12,kernel_size=3,stride=2),
        #     torch.nn.SiLU(),
        #     torch.nn.Conv2d(12,10,2,bias=False),
        #     # torch.nn.Flatten()
        # )

    def forward(self,x):
        # return self._layer(x)
        h0 = self._input_layer(x)
        h = self._layer_01(h0)+h0
        h = self._max_pool(h)
        h = self._output_layer(h)
        return h



if __name__ == '__main__':

    x = torch.randn(3,1,28,28)
    net = MnistNet()
    y = net(x)
    print(y.shape)