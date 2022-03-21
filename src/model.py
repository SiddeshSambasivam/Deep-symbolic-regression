from turtle import forward
import torch
import typing as T
from .functions import Function, count_double_inputs, default_funcs
from .utils import IncorrectMatrixDimension

class SymbolicLayer(torch.nn.Module):
    """Represents a fully connected NN with custom activation functions"""

    def __init__(self, input_size, funcs: T.List[Function]=default_funcs, init_std:torch.float=0.1):

        super(SymbolicLayer, self).__init__()
        self.output:torch.Tensor = None

        self.funcs = [fn.torch for fn in funcs]
        self.n_funcs = len(self.funcs)
        self.n_double = count_double_inputs(funcs=funcs)
        self.n_single = self.n_funcs - self.n_double

        self.input_size = input_size
        self.output_size = self.n_double + self.n_funcs # (2*self.n_double ) + (self.n_funcs-self.n_double)

        self.W = torch.normal(0.0, init_std, (self.input_size, self.output_size)) 

    def forward(self, x):
        
        y = torch.matmul(x, self.W) # ?xn X nxp => ?xp
        self.output = []
        if y.shape != (x.shape[0], self.output_size):
            raise IncorrectMatrixDimension((x.shape[0], self.output_size), y.shape) 

        fn_idx = 0 # tracks the function to apply
        idx = 0
        while fn_idx < self.n_single:
            self.output.append(self.funcs[fn_idx](y[:, idx]))
            idx += 1
            fn_idx += 1
        
        while fn_idx < self.n_funcs:
            self.output.append(self.funcs[fn_idx](y[:, idx], y[:, idx+1]))
            fn_idx += 1
            idx + 2

        self.output = torch.stack(self.output, dim=1) # concatenates along the column => ?xn_funcs

        return self.output

    def get_weights(self):
        return self.W.cpu().detach().numpy()

class SymbolicNN(torch.nn.Module):
    """A NN which has SymbolicLayer as hidden layers and produces one output"""

    def __init__(self, depth:int, funcs: T.Type[Function], init_std:torch.float=0.1) -> None:

        super(SymbolicNN, self).__init__()
        
        self.funcs = funcs
        self.std = init_std
        self.n_layer_inp = [1] + [len(funcs)]*depth # Each feature has its own NN, so input has only one feature

        self.layers = [SymbolicLayer(self.n_layer_inp[i], self.funcs, self.std) for i in range(depth)]
        self.out_layer = torch.nn.Parameter(torch.randn(self.n_layer_inp[-1], 1))

        self.hidden_layers = torch.nn.Sequential(*self.layers)
    
    def forward(self, x) -> torch.Tensor:
        z = self.hidden_layers(x)
        out = torch.matmul(z, self.out_layer)
        return out        

    def get_weights(self) -> torch.Tensor:
        """Returns the weights of the NN"""
        return [layer.get_weights() for layer in self.hidden_layers] + [self.out_layer.cpu().detach().numpy()]