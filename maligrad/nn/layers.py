import numpy as np

from maligrad.autograd.engine import DataNode
from .engine import Module, Parameter


class Linear(Module):

    def __init__(self, in_features: int,
                 out_features: int, bias: bool = True) -> None:
        self.W = Parameter(data=np.zeros((in_features, out_features)))
        self.b = Parameter(data=np.zeros((out_features, 1))) if bias else 0

    def forward(self, x: DataNode) -> DataNode:
        return x @ self.W + self.b
