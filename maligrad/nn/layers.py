from typing import Any
import numpy as np

from maligrad.autograd.engine import DataNode
from maligrad.nn.engine import Module, Parameter
import maligrad.nn.functional as F

class Linear(Module):

    def __init__(self, in_features: int,
                 out_features: int, bias: bool = True) -> None:
        self.W = Parameter(data=np.zeros((in_features, out_features)))
        self.b = Parameter(data=np.zeros((out_features, 1))) if bias else 0

    def forward(self, x: DataNode) -> DataNode:
        return x @ self.W + self.b

class ReLU(Module):

    def forward(self, x: DataNode) -> DataNode:
        return F.relu(x)
    
class Sigmoid(Module):

    def forward(self, x: DataNode) -> DataNode:
        return F.sigmoid(x)

class Softmax(Module):

    def forward(self, x: DataNode, axis: int = -1) -> DataNode:
        return F.softmax(x, axis)