import numpy as np
from numpy import ndarray

from typing import Any, Callable, Union
import operator
from abc import ABC

def accumulate_grad(grad: ndarray, partial: ndarray) -> None:
    # sum leads if partial bigger than grad
    if (dim_diff := partial.ndim - grad.ndim) > 0: 
        partial = partial.sum(tuple(range(dim_diff)))
    # right-align, sum mismatched axis in partial
    for i, (d_g, d_p) in enumerate(zip(grad.shape[::-1], partial.shape[::-1])):
        if d_g == 1 and d_p > 1:
            partial = partial.sum(-i-1, keepdims=True)
    grad += partial


###############################
# Similar to Variable in torch
###############################

class Variable:

    # so numpy knows not to promote to ndarray
    # 1337 just because I'm a leet hackerman
    __array_priority__ = 1337

    def __array__(self):
        return self.data
    
    def __init__(self, data, requires_grad = False, _src_fcn = None, _src_index = 0):
        self.data = np.array(data)
        self.grad = None
        self.requires_grad = requires_grad
        self.src_fcn = self if requires_grad and _src_fcn is None else _src_fcn
        self.src_index = _src_index

    def accumulate_grad(self, partial, index):
        # construct gradient if not present and update it
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        accumulate_grad(self.grad, partial)

    def backward(self, partial: ndarray | None = None):
        # check if backprop can be started
        err_msg = "Cannot start backprop {reason}"
        assert self.requires_grad, err_msg.format(reason="from Variable that does not require grad.")
        if partial is None:
            assert self.size == 1, err_msg.format(reason="from non-singleton Variable.")
            partial = np.ones_like(self.data)
        
        # deposit grad into source fcn, toposort compute graph
        # and propagate grads through each function node
        self.src_fcn.accumulate_grad(partial, self.src_index)
        for fcn in reversed(self.src_fcn.toposort()):
            fcn.propagate_grads()

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype

    @classmethod
    def ensure(cls, other):
        if isinstance(other, cls):
            return other
        return cls(other)
    
     # OP OVERRIDES

    def __add__(self, other: "Variable" | Any) -> "Variable":
        add_node = Add()
        return add_node(self, other)
    
    def __radd__(self, other: "Variable" | Any) -> "Variable":
        return self + other
    
    def __mul__(self, other: "Variable" | Any) -> "Variable":
        mul_node = Mul()
        return mul_node(self, other)
    
    def __rmul__(self, other: "Variable" | Any) -> "Variable":
        return self * other
    
    def __neg__(self) -> "Variable":
        return (-1) * self

    def __sub__(self, other: "Variable" | Any) -> "Variable":
        return -other + self
    
    def __rsub__(self, other: "Variable" | Any) -> "Variable":
        return -self + other
    
    def __matmul__(self, other: "Variable" | Any) -> "Variable":
        other = self.ensure(other)
        matmul_node = Matmul()
        return matmul_node(self, other)
    
    def __rmatmul__(self, other: "Variable" | Any) -> "Variable":
        other = self.ensure(other)
        matmul_node = Matmul()
        return matmul_node(other, self)
    
    def __pow__(self, other: "Variable" | Any) -> "Variable":
        other = self.ensure(other)
        pow_node = Pow()
        return pow_node(self, other)
    
    def __rpow__(self, other: "Variable" | Any) -> "Variable":
        other = self.ensure(other)
        pow_node = Pow()
        return pow_node(other, self)
    
    def __truediv__(self, other: "Variable" | Any) -> "Variable":
        return self * other ** -1
    
    def __rtruediv__(self, other: "Variable" | Any) -> "Variable":
        return self ** -1 * other
    
    def __gt__(self, other: "Variable" | Any) -> "Variable":
        compare_node = Compare(operator.gt)
        return compare_node(self, other)
    
    def __ge__(self, other: "Variable" | Any) -> "Variable":
        compare_node = Compare(operator.ge)
        return compare_node(self, other)
    
    def __lt__(self, other: "Variable" | Any) -> "Variable":
        compare_node = Compare(operator.lt)
        return compare_node(self, other)
    
    def __le__(self, other: "Variable" | Any) -> "Variable":
        compare_node = Compare(operator.le)
        return compare_node(self, other)
    
    def __eq__(self, other: "Variable" | Any) -> "Variable":
        compare_node = Compare(operator.eq)
        return compare_node(self, other)
    
    def __ne__(self, other: "Variable" | Any) -> "Variable":
        compare_node = Compare(operator.ne)
        return compare_node(self, other)
    
    def __getitem__(self, slices: Union[slice, list[slice]]) -> "Variable":
        slice_node = Slice()
        return slice_node(self, slices)
    
    def __setitem__(self, slices: Union[slice, list[slice]]) -> "Variable":
        # TODO in the far far future
        raise NotImplementedError("Assignment currently not supported.") 
    
    def __neg__(self) -> "Variable":
        return self * -1
    
    def __pos__(self) -> "Variable":
        return self
    
    def __invert__(self) -> "Variable":
        invert_node = Invert()
        return invert_node(self)
    
    def __str__(self) -> str:
        return f"""Variable(\ndata=\n{self.data},\ngrad=\n{self.grad}\n)"""
    
    def __repr__(self) -> str:
        return str(self)
    
    def __hash__(self) -> str:
        return id(self)
    
    # Useful ops from numpy
    
    def reshape(self, *new_shape: Union[list[int], tuple[int]]) -> "Variable":
        reshape_node = Reshape()
        return reshape_node(self, *new_shape)
    
    def transpose(self, axes: Union[list[int], tuple[int]] | None = None) -> "Variable":
        transpose_node = Transpose()
        return transpose_node(self, axes)
    
    def swap_axes(self, axis1: int, axis2: int) -> "Variable":
        to_swap = [axis1, axis2]
        axes = np.arange(self.ndim)
        axes[to_swap] = axes[to_swap[::-1]]
        transpose_node = Transpose()
        return transpose_node(self, axes)
        
    def sum(self, axis: Union[int, list[int], tuple[int], None] = None, keepdims: bool = False) -> "Variable":
        sum_node = Sum()
        return sum_node(self, axis, keepdims)

    def mean(self, axis: Union[int, list[int], tuple[int], None] = None, keepdims: bool = False) -> "Variable":
        if axis is None:
            axis = np.arange(self.ndim)
        sum = self.sum(axis, keepdims)
        n = np.array(self.shape)[list(axis)].prod()
        mean = 1 / n * sum
        return mean
    
    def max(self, axis: Union[int, list[int], tuple[int], None] = None, keepdims: bool = False) -> "Variable":
        sum_node = Max()
        return sum_node(self, axis, keepdims)
    
    def min(self, axis: Union[int, list[int], tuple[int], None] = None, keepdims: bool = False) -> "Variable":
        return -(-self).max(axis, keepdims)

    def maximum(self, other: "Variable" | Any) -> "Variable":
        mask = self >= other
        return mask * self + ~mask * other
    
    def minimum(self, other: "Variable" | Any) -> "Variable":
        mask = self <= other
        return mask * self + ~mask * other
    
    def squeeze(self, axes: Union[int, list[int], tuple[int]] | None = None) -> "Variable":
        new_shape = np.squeeze(self.data, axes).shape
        return self.reshape(new_shape)

    def unsqueeze(self, axes: Union[int, list[int], tuple[int]] | None = None) -> "Variable":
        new_shape = np.expand_dims(self.data, axes).shape # this is kinda funny
        return self.reshape(new_shape)


################################
# Function here is if you joined
# torch's Function and context
################################
    
class Function(ABC):
    previous: list[tuple["Function", int]] # functions and indexes of return that generated input Variables
    grad_buffer: list # gradient buffer for each of the outputs
    grad_buffer_info: list # shapes and dtypes of such buffers - so buffers are constructed only when needed
    saved_ctx: list # assets saved during forward that are needed for backward

    is_differentiable = True

    def __init__(self) -> None:
        self.saved_ctx = []

    def __call__(self, *inputs: Any) -> Any:
        # setup
        self.previous = [
            (i.src_fcn, i.src_index) if isinstance(i, Variable) else (None, None) 
            for i in inputs]
        requires_grad = self.is_differentiable and\
            any(i.requires_grad for i in inputs if isinstance(i, Variable))
        forward_data = [i.data if isinstance(i, Variable) else i for i in inputs]

        # propagate inputs through self and construct Vars
        outputs = [Variable(
            data=data,
            requires_grad=requires_grad,
            _src_fcn=self if requires_grad else None,
            _src_index=i if requires_grad else 0
        ) for i, data in enumerate(self.forward(*forward_data))]

        # save buffer info and clear buffers if full
        self.grad_buffer_info = [
            (v.shape, v.dtype) if isinstance(v, Variable) else None
            for v in outputs]
        self.grad_buffer = None
        return outputs if len(outputs) > 1 else outputs[0]
    
    def accumulate_grad(self, partial, index):
        if self.grad_buffer is None:
            self.grad_buffer = [
                np.zeros(shape, dtype) 
                for shape, dtype in self.grad_buffer_info]
        accumulate_grad(self.grad_buffer[index], partial)
    
    def propagate_grads(self):
        # update collected partial gradients through self
        updated_partials = self.backward(*self.grad_buffer)

        # deposit updated partial gradients into previous buffers/variables
        for (prev, i), partial in zip(self.previous, updated_partials):
            if isinstance(prev, (Function, Variable)):
                prev.accumulate_grad(partial, i)
        
        # clear buffers
        self.grad_buffer = None
        self.grad_buffer_info = None
    
    def save_for_backprop(self, *to_save):
        self.saved_ctx += to_save

    def forward(self, *inputs):
        raise NotImplementedError
    
    def backward(self, *partials):
        raise NotImplementedError
    
    def toposort(self, visited: set | None = None) -> list["Function"]:
        if visited is None:
            visited = set()
        out = []
        if self not in visited:
            visited.add(self)
            for p, _ in self.previous:
                if not isinstance(p, Function): continue 
                out += p.toposort(visited)
            out.append(self)
        return out

######
# OPS
######
    
class Add(Function):

    def forward(self, a, b):
        return a + b,
    
    def backward(self, partial):
        return partial, partial
    

class Mul(Function):

    def forward(self, a: ndarray, b: ndarray) -> ndarray:
        self.save_for_backprop(a, b)
        return a * b,
    
    def backward(self, partial: ndarray) -> tuple[ndarray]:
        a, b = self.saved_ctx
        return b * partial, a * partial


class Pow(Function):

    def forward(self, base: ndarray, exp: ndarray) -> ndarray:
        self.save_for_backprop(base, exp)
        return base ** exp,
    
    def backward(self, partial: ndarray) -> tuple[ndarray]:
        base, exp = self.saved_ctx
        return (exp * base ** (exp - 1) * partial,
            base ** exp * np.log(base) * partial) # TODO: fix warning when log <= 0


class Matmul(Function):

    def forward(self, a: ndarray, b: ndarray) -> ndarray:
        self.save_for_backprop(a, b)
        return a @ b,
    
    def backward(self, partial: ndarray) -> tuple[ndarray]:
        a, b = self.saved_ctx
        return partial @ b.swapaxes(-2, -1), a.swapaxes(-2, -1) @ partial


class Slice(Function):

    def forward(self, data: ndarray, slices: Any) -> ndarray:
        self.save_for_backprop(slices, data.shape, data.dtype)
        return data[slices], 
    
    def backward(self, partial: ndarray) -> tuple[ndarray]:
        slices, original_shape, original_dtype = self.saved_ctx
        template = np.zeros(original_shape, original_dtype)
        np.add.at(template, slices, partial)
        return template, None


class Compare(Function):

    is_differentiable = False

    def __init__(self, compare_op: Callable) -> None:
        super().__init__()
        self.op = compare_op

    def forward(self, a: ndarray, b: ndarray | Any) -> ndarray:
        return self.op(a, b),
    

class Invert(Function):

    is_differentiable = False

    def forward(self, a: ndarray) -> ndarray:
        return ~a,


class Reshape(Function):

    def forward(self, data: ndarray, *new_shape: tuple[int]) -> ndarray:
        self.save_for_backprop(data.shape)
        return data.reshape(*new_shape),
    
    def backward(self, partial: ndarray) -> tuple[ndarray]:
        (old_shape, ) = self.saved_ctx
        return (partial.reshape(old_shape), None)


class Sum(Function):
    
    def forward(self, data: ndarray, axis: int| tuple | None, keepdims: bool) -> ndarray:
        if axis is None:
            axis = tuple(range(data.ndim))
        self.save_for_backprop(axis, keepdims)
        return np.sum(data, axis=axis, keepdims=keepdims),
    
    def backward(self, partial: ndarray) -> tuple[ndarray]:
        axis, keepdims = self.saved_ctx
        if keepdims:
            return (partial, )
        return (np.expand_dims(partial, axis=axis), None, None)
    

class Max(Function):

    def forward(self, data: ndarray, axis: int | tuple | None, keepdims: bool) -> tuple[ndarray]:
        if axis is None:
            axis = tuple(range(data.ndim))
        max_ = data.max(axis=axis, keepdims=True)
        idxs = data == max_
        self.save_for_backprop(idxs, data.shape, data.dtype)
        return (max_,) if keepdims else (max_.squeeze(axis),)
    
    def backward(self, partial: ndarray) -> tuple[ndarray]:
        idxs, original_shape, original_dtype = self.saved_ctx
        template = np.zeros(original_shape, original_dtype)
        template[idxs] += partial.flatten()
        return (template, None, None)


class Transpose(Function):
    
    def forward(self, data: ndarray, axes: list[int] | tuple[int] | None) -> ndarray:
        if axes is None:
            axes = tuple(range(data.ndim))[::-1]
        self.save_for_backprop(np.argsort(axes))
        return data.transpose(axes), 
    
    def backward(self, partial: ndarray) -> tuple[ndarray]:
        (axes_inv, ) = self.saved_ctx
        return (partial.transpose(axes_inv), None)