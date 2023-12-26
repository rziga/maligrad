from typing import Any, List, Iterable
from abc import ABC, abstractmethod
from numpy import ndarray

from ..autograd.engine import DataNode


def search_parameters(item: Any) -> List["Parameter"]:
    if isinstance(item, Parameter):
        return [item]
    elif isinstance(item, Module):
        iterator = vars(item).values()
    elif isinstance(item, Iterable):
        iterator = item
    else:
        return []
    return [p for item in iterator for p in search_parameters(item)]

def search_submodules(module: "Module") -> List["Module"]:
    submodules = []
    for var in vars(module).values():
        if isinstance(var, Module):
            submodules += [var]
        if isinstance(var, Iterable):
            submodules += [el for el in var if isinstance(el, Module)]
    return submodules


class Parameter(DataNode):

    def __init__(self, data: ndarray | Any) -> None:
        super().__init__(data, requires_grad=True)


class Module(ABC):

    def __call__(self, *inputs: DataNode | Any) -> DataNode:
        return self.forward(*inputs)

    @abstractmethod
    def forward(self, *inputs: DataNode | Any) -> DataNode:
        raise NotImplementedError
    
    def parameters(self) -> List[Parameter]:
        return search_parameters(self)

    def submodules(self) -> List["Module"]:
        return search_submodules(self)
