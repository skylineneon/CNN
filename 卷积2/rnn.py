import torch
from torch import nn

class RNNCell(nn.Module):

    def __init__(self,input_dim,bias=True):
        super().__init__()

        self._layer = nn.Linear(input_dim,input_dim,bias)

    def forward(self,x,h):
        _h = x + h
        return self._layer(_h)
    
class RNNLayer(nn.Module):

    def __init__(self,input_dim):
        super().__init__()

        # self._layers = nn.ModuleList(
        #     [RNNCell(input_dim) for _ in range(num_layers)]
        # )

        self.rnncell= RNNCell(input_dim)

    def forward(self,xs,h):
        """
        @param: xs NSV
        @param: h NV
        """
        _outputs = []
        _h = h
        for _x in range(xs.shape[1]):
            _h = self.rnncell(_x,_h)
            _outputs.append(_h)
        
        return _outputs,_outputs[-1]

class RNNNet(nn.Module):

    def __init__(self,num_layers,input_dim):
        super().__init__()

        self._layers = nn.ModuleList(
            [RNNLayer(input_dim) for _ in range(num_layers)]
        )

    def forward(self,xs,h):
        _h = h

        for _layer in self._layers:
            _os,_h = _layer(xs,_h)
        
        return _h

