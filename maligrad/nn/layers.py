from typing import Any
import numpy as np

from maligrad.autograd.engine import Variable
from maligrad.nn.engine import Module, Parameter
import maligrad.nn.functional as F


class Linear(Module):

    def __init__(self, in_features: int,
                 out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.W = Parameter(data=np.zeros((in_features, out_features)))
        self.b = Parameter(data=np.zeros(out_features)) if bias else 0

    def forward(self, x: Variable) -> Variable:
        return x @ self.W + self.b

class ReLU(Module):

    def forward(self, x: Variable) -> Variable:
        return F.relu(x)
    
class Sigmoid(Module):

    def forward(self, x: Variable) -> Variable:
        return F.sigmoid(x)

class Softmax(Module):

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: Variable) -> Variable:
        return F.softmax(x, self.axis)
    
class Conv(Module):

    def __init__(self, 
                 in_chan: int , out_chan: int, ker_shape: tuple, 
                 stride: int | tuple = 1, dilation: int | tuple = 1, 
                 bias: bool = True) -> None:
        super().__init__()
        self.ker = Parameter(data=np.zeros((out_chan, in_chan, *ker_shape)))
        self.b   = Parameter(data=np.zeros((out_chan, *([1] * len(ker_shape))))) if bias else 0
        self.stride, self.dilation, self.dim = stride, dilation, len(ker_shape)+1

    def forward(self, x: Variable) -> Variable:
        return F.conv(
            x, self.ker, self.dim,
            self.stride, self.dilation
            ).squeeze(-self.dim) + self.b