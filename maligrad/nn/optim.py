from abc import ABC, abstractmethod
from typing import List

from maligrad.nn.engine import Parameter


class Optimizer(ABC):

    def __init__(self, parameters: List[Parameter], lr: float) -> None:
        self.parameters = parameters
        self.lr = lr

    @abstractmethod
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad[:] = 0.


class SGD(Optimizer):

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad