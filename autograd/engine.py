import numpy as np
from numpy import ndarray # for typing only

from typing import List, Union, Set, Any, Tuple
from abc import abstractmethod


class Node:

    def __init__(self, children: List[Union["Node", None]] = []) -> None:
        self._children = children

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
    

##########################
# DataNode
##########################

class DataNode(Node):

    def __init__(self, data: Union[ndarray, Any], requires_grad: bool = False) -> None:
        super().__init__([]) # [origin FunctionNode] - [] for natural init
        self.data = data if isinstance(data, ndarray) else np.array(data)
        self.grad = np.zeros_like(self.data) if requires_grad else None

    def backward(self, partial: ndarray | None = None) -> None:
        # start backprop
        if partial is None:
            assert self.data.squeeze().ndim == 0, "bacprop without partial grad can be only started from scalars"
            partial = 1.
        self.grad += partial
        graph_nodes = [n for n in reversed(self.toposort()) if isinstance(n, FunctionNode)]
        for node in graph_nodes:
            node.backward()

    def accumulate_grad(self, grad: ndarray) -> None:
        if self.grad.shape == grad.shape:
            self.grad += grad
        else:
            # bug if broadcast axis between non broadcast axes
            #self.grad += grad.reshape(-1, *self.grad.shape).sum(axis=0)
            shg = grad.shape
            shs = [1]*(grad.ndim - self.grad.ndim) + list(self.grad.shape)
            to_sum = tuple(np.argwhere(np.not_equal(shg, shs)).squeeze())
            self.grad += grad.sum(axis=to_sum).squeeze()
    
    @staticmethod
    def promote(data: "DataNode" | Any):
        if isinstance(data, DataNode):
            return data
        return DataNode(data)

    @property
    def requires_grad(self) -> bool:
        return self.grad is not None
    
    @property
    def shape(self) -> List[int]:
        return self.data.shape
    
    # OP OVERRIDES

    def __add__(self, other: "DataNode" | Any) -> "DataNode":
        other = self.promote(other)
        add_node = Add()
        return add_node(self, other)
    
    def __mul__(self, other: "DataNode" | Any) -> "DataNode":
        other = self.promote(other)
        mul_node = Mul()
        return mul_node(self, other)
    
    def __matmul__(self, other: "DataNode" | Any) -> "DataNode":
        other = self.promote(other)
        matmul_node = Matmul()
        return matmul_node(self, other)
    
    def __pow__(self, other: "DataNode" | Any) -> "DataNode":
        other = self.promote(other)
        pow_node = Pow()
        return pow_node(self, other)
    
    def __truediv__(self, other: "DataNode" | Any) -> "DataNode":
        other = self.promote(other)
        return self * other ** -1
    
    def __getitem__(self, slices: Union[slice, List[slice]]) -> "DataNode":
        slice_node = Slice()
        return slice_node(self, slices)
    

##########################
# FunctionNode
##########################

class FunctionNode(Node):

    def __init__(self) -> None:
        super().__init__([]) # inputs to the Function
        self.results = [] # results of applying Function on inputs
        self.backprop_assets = [] # saved stuff for efficient backward in Function

    def __call__(self, *inputs: DataNode | Any) -> DataNode:
        return self.forward(*inputs)
    
    def forward(self, *inputs: DataNode | Any) -> DataNode:
        # TODO: handle broadcasting in here if needed -> HOW DO U KNOW? IDK BRO 😭 
        # -> WAIT I'M GOATED: DO YOU EVEN NEED TO TRACK SHAPES IF THE ACCUMULATE GRAD DOES IT'S JOB ??? IDK BRO WE TRY

        # attach inputs
        input_data_nodes = [i for i in inputs if isinstance(i, DataNode)]
        self.add_children(*input_data_nodes)

        # create new DataNode and add itself as its source
        data = self._forward(*[i.data if isinstance(i, DataNode) else i for i in inputs])
        new_data_node = DataNode(data, requires_grad=any(i.requires_grad for i in input_data_nodes))
        if not self._is_differentiable:
            return new_data_node # result detached from current node
        new_data_node.add_children(self)

        # attach created DataNode as result
        self.attach_results(new_data_node)
        return new_data_node
    
    def backward(self):
        grads = self._backward(*[t.grad for t in self.results])
        for c, grad in zip(self.children, grads):
            if c.requires_grad:
                c.accumulate_grad(grad)

    @abstractmethod
    def _forward(self, *inputs: ndarray | Any) -> ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def _backward(self, *partials: ndarray) -> Tuple[ndarray]:
        raise NotImplementedError

    @property
    def _is_differentiable(self):
        return True
    
    def attach_results(self, *results: DataNode) -> None:
        self.results += results

    def save_for_backprop(self, *to_save: Any) -> None:
        self.backprop_assets += to_save


##########################
# Function Definitions
##########################

class Add(FunctionNode):

    def _forward(self, a: ndarray, b: ndarray) -> ndarray:
        return a + b
    
    def _backward(self, partial: ndarray) -> Tuple[ndarray]:
        return partial, partial
    
class Mul(FunctionNode):

    def _forward(self, a: ndarray, b: ndarray) -> ndarray:
        self.save_for_backprop(a, b)
        return a * b
    
    def _backward(self, partial: ndarray) -> Tuple[ndarray]:
        a, b = self.backprop_assets
        return b * partial, a * partial

class Pow(FunctionNode):

    def _forward(self, base: ndarray, exp: ndarray) -> ndarray:
        self.save_for_backprop(base, exp)
        return base ** exp
    
    def _backward(self, partial: ndarray) -> Tuple[ndarray]:
        base, exp = self.backprop_assets
        return exp * base ** (exp - 1) * partial, base ** exp * np.log(base) * partial

class Matmul(FunctionNode):

    def _forward(self, a: ndarray, b: ndarray) -> ndarray:
        self.save_for_backprop(a, b)
        return a @ b
    
    def _backward(self, partial: ndarray) -> Tuple[ndarray]:
        a, b = self.backprop_assets
        print(a.shape, b.shape, partial.shape)
        return partial @ b.swapaxes(-2, -1), a.swapaxes(-2, -1) @ partial
    
class Slice(FunctionNode):

    def _forward(self, data: ndarray, slices: Any) -> ndarray:
        self.save_for_backprop(slices, data.shape)
        return data[slices]
    
    def _backward(self, partial: ndarray) -> Tuple[ndarray]:
        slices, original_shape = self.backprop_assets
        template = np.zeros(original_shape)
        template[slices] += partial
        return (template, )
    
class Reshape(FunctionNode):

    def _forward(self, data: ndarray, new_shape: Tuple[int]) -> ndarray:
        self.save_for_backprop(data.shape)
        return data.reshape(new_shape)
    
    def _backward(self, partial: ndarray) -> Tuple[ndarray]:
        old_shape = self.backprop_assets
        return (partial.reshape(old_shape), )
    