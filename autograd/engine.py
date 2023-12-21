import numpy as np
from numpy import ndarray # for typing only

import operator
from typing import List, Union, Set, Any, Tuple
from abc import ABC

from .ops import *


class Node(ABC):

    def __init__(self, children: List["Node"] | None = None) -> None:
        self._children = children if children is not None else []

    @property
    def children(self) -> List["Node"]:
        return self._children
    
    def add_children(self, *children: "Node") -> None:
        self._children += children

    @property
    def is_leaf(self) -> bool:
        return bool(self._children)
    
    def toposort(self, visited: Set = set()) -> List["Node"]:
        out = []
        if self not in visited:
            visited.add(self)
            for child in self.children:
                out += child.toposort()
            out.append(self)
        return out
    

##################################
# DataNode - liek tensors in torch
##################################

class DataNode(Node):

    __array_priority__ = 1337 # so numpy knows not to promote to ndarray, 1337 just because I'm a leet hackerman

    def __init__(self, data: Union[ndarray, Any], requires_grad: bool = False) -> None:
        super().__init__() # [origin FunctionNode] - [] for natural init
        self.data = data if isinstance(data, ndarray) else np.array(data)
        self.grad = np.zeros_like(self.data) if requires_grad else None

    def backward(self, partial: ndarray | None = None) -> None:
        assert self.grad is not None,\
            "Cannot backpropagate from DataNode with disabled gradient tracking."
        # start backprop
        if partial is None:
            assert self.data.squeeze().ndim == 0,\
                "Backprop without seed partial gradient can be only started from scalars."
            partial = np.array(1.)
        self.grad += partial
        graph_nodes = [n for n in reversed(self.toposort()) if isinstance(n, FunctionNode)]
        for node in graph_nodes:
            node.backward()

    def accumulate_grad(self, grad: ndarray) -> None:
        if self.grad.shape == grad.shape:
            self.grad += grad
        else:
            # right align shapes
            shape_in = grad.shape
            lead = (1,) * (grad.ndim - self.grad.ndim)
            shape_self = lead + self.grad.shape

            # sum mismatched axis
            shape_mismatches = np.argwhere(np.not_equal(shape_in, shape_self)).squeeze()
            axs_to_sum = tuple(shape_mismatches) if shape_mismatches.ndim != 0 else shape_mismatches
            grad = grad.sum(axis=axs_to_sum, keepdims=True)
            self.grad += grad.squeeze(tuple(range(len(lead)))) # squeeze leading 1s
    
    @staticmethod
    def promote(data: "DataNode" | Any) -> "DataNode":
        if isinstance(data, DataNode):
            return data
        return DataNode(data)

    @property
    def requires_grad(self) -> bool:
        return self.grad is not None
    
    @property
    def shape(self) -> List[int]:
        return self.data.shape
    
    @property
    def size(self) -> int:
        return self.data.size
    
    @property
    def ndim(self) -> int:
        return self.data.ndim
    
    # OP OVERRIDES

    def __add__(self, other: "DataNode" | Any) -> "DataNode":
        add_node = FunctionNode(Add())
        return add_node(self, other)
    
    def __radd__(self, other: "DataNode" | Any) -> "DataNode":
        return self + other
    
    def __mul__(self, other: "DataNode" | Any) -> "DataNode":
        mul_node = FunctionNode(Mul())
        return mul_node(self, other)
    
    def __rmul__(self, other: "DataNode" | Any) -> "DataNode":
        return self * other
    
    def __matmul__(self, other: "DataNode" | Any) -> "DataNode":
        other = self.promote(other)
        matmul_node = FunctionNode(Matmul())
        return matmul_node(self, other)
    
    def __rmatmul__(self, other: "DataNode" | Any) -> "DataNode":
        other = self.promote(other)
        matmul_node = FunctionNode(Matmul())
        return matmul_node(other, self)
    
    def __pow__(self, other: "DataNode" | Any) -> "DataNode":
        other = self.promote(other)
        pow_node = FunctionNode(Pow())
        return pow_node(self, other)
    
    def __rpow__(self, other: "DataNode" | Any) -> "DataNode":
        other = self.promote(other)
        pow_node = FunctionNode(Pow())
        return pow_node(other, self)
    
    def __truediv__(self, other: "DataNode" | Any) -> "DataNode":
        return self * other ** -1
    
    def __rtruediv__(self, other: "DataNode" | Any) -> "DataNode":
        return self ** -1 * other
    
    def __gt__(self, other: "DataNode" | Any) -> "DataNode":
        compare_node = FunctionNode(Compare(operator.gt))
        return compare_node(self, other)
    
    def __ge__(self, other: "DataNode" | Any) -> "DataNode":
        compare_node = FunctionNode(Compare(operator.ge))
        return compare_node(self, other)
    
    def __lt__(self, other: "DataNode" | Any) -> "DataNode":
        compare_node = FunctionNode(Compare(operator.lt))
        return compare_node(self, other)
    
    def __le__(self, other: "DataNode" | Any) -> "DataNode":
        compare_node = FunctionNode(Compare(operator.le))
        return compare_node(self, other)
    
    def __eq__(self, other: "DataNode" | Any) -> "DataNode":
        compare_node = FunctionNode(Compare(operator.eq))
        return compare_node(self, other)
    
    def __ne__(self, other: "DataNode" | Any) -> "DataNode":
        compare_node = FunctionNode(Compare(operator.ne))
        return compare_node(self, other)
    
    def __getitem__(self, slices: Union[slice, List[slice]]) -> "DataNode":
        slice_node = FunctionNode(Slice())
        print(slices)
        return slice_node(self, slices)
    
    def __setitem__(self, slices: Union[slice, List[slice]]) -> "DataNode":
        # TODO
        raise NotImplementedError("assignment currently not supported") 
    
    def __neg__(self) -> "DataNode":
        return self * -1
    
    def __pos__(self) -> "DataNode":
        return self
    
    def __invert__(self) -> "DataNode":
        invert_node = FunctionNode(Invert())
        return invert_node(self)
    
    def __str__(self) -> str:
        return f"""DataNode(\ndata=\n{self.data},\ngrad=\n{self.grad}\n)"""
    
    def __repr__(self) -> str:
        return str(self)
    
    def __hash__(self) -> str:
        return id(self)
    
    # Useful ops from numpy
    
    def reshape(self, *new_shape: Union[List[int], Tuple[int]]) -> "DataNode":
        reshape_node = FunctionNode(Reshape())
        return reshape_node(self, new_shape)
    
    def transpose(self, axes: Union[List[int], Tuple[int]] | None = None) -> "DataNode":
        transpose_node = FunctionNode(Transpose())
        return transpose_node(self, axes)
    
    def swap_axes(self, axis1: int, axis2: int) -> "DataNode":
        to_swap = [axis1, axis2]
        axes = np.arange(self.ndim)
        axes[to_swap] = axes[to_swap[::-1]]
        transpose_node = FunctionNode(Transpose())
        return transpose_node(self, axes)
        
    def sum(self, axis: Union[int, List[int], Tuple[int], None] = None, keepdims: bool = False):
        sum_node = FunctionNode(Sum())
        return sum_node(self, axis, keepdims)
    
    def maximum(self, other: "DataNode" | Any) -> "DataNode":
        mask = self >= other
        return mask * self + ~mask * other
    
    def minimum(self, other: "DataNode" | Any) -> "DataNode":
        mask = self <= other
        return mask * self + ~mask * other
    
    def squeeze(self, axes: Union[int, List[int], Tuple[int]] | None = None) -> "DataNode":
        # TODO
        pass

    def unsqueeze(self, axes: Union[int, List[int], Tuple[int]] | None = None) -> "DataNode":
        # TODO
        raise NotImplementedError
        

##########################
# FunctionNode
##########################

class FunctionNode(Node):

    def __init__(self, function: Function) -> None:
        super().__init__() # inputs to the Function
        self.fcn = function

        self.results: List[DataNode] = [] # results of applying Function on inputs
        self.backprop_assets: List[Any] = [] # saved context for efficient backward in Function

    def __call__(self, *inputs: DataNode | Any) -> DataNode:
        return self.forward(*inputs)
    
    def forward(self, *inputs: DataNode | Any) -> DataNode:
        # TODO: handle broadcasting in here if needed -> HOW DO U KNOW? IDK BRO 😭 
        # -> WAIT I'M GOATED: DO YOU EVEN NEED TO TRACK SHAPES IF 
        # THE ACCUMULATE GRAD DOES IT'S JOB ??? IDK BRO WE TRY

        # attach inputs
        input_data_nodes = [i for i in inputs if isinstance(i, DataNode)]
        self.add_children(*input_data_nodes)

        # get data for new DataNode
        forward_inputs = [
            i.data if isinstance(i, DataNode) else i
            for i in inputs]
        data = self.fcn.forward(self, *forward_inputs)

        # result detached from computation graph
        if not self.fcn.is_differentiable or\
            not any(i.requires_grad for i in input_data_nodes):
            new_data_node = DataNode(data, requires_grad=False)
            return new_data_node
        
        # add self (function node) as child of the new data node
        new_data_node = DataNode(data, requires_grad=True)
        new_data_node.add_children(self)

        # attach created DataNode as result
        self.attach_results(new_data_node)
        return new_data_node
    
    def backward(self):
        if not self.fcn.is_differentiable:
            raise ValueError("backward cannot be used on non-differentiable FunctionNode")
        grads = self.fcn.backward(self, *[t.grad for t in self.results])
        for c, grad in zip(self.children, grads):
            if c.requires_grad:
                c.accumulate_grad(grad)
    
    def attach_results(self, *results: DataNode) -> None:
        self.results += results

    def save_for_backprop(self, *to_save: Any) -> None:
        self.backprop_assets += to_save