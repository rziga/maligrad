from abc import abstractmethod, ABC
from typing import Tuple, List, Callable, Any

import numpy as np
from numpy import ndarray


class Function(ABC):

    @property
    def is_differentiable(self):
        return True

    @abstractmethod
    def forward(self, ctx: "FunctionNode", *inputs: Tuple[ndarray | Any]) -> Tuple[ndarray]:
        raise NotImplementedError

    def backward(self, ctx: "FunctionNode", *partials: Tuple[ndarray]) -> Tuple[ndarray]:
        raise NotImplementedError


class Add(Function):

    def forward(self, ctx, a: ndarray, b: ndarray) -> ndarray:
        return a + b
    
    def backward(self, ctx, partial: ndarray) -> Tuple[ndarray]:
        return partial, partial
    
class Mul(Function):

    def forward(self, ctx, a: ndarray, b: ndarray) -> ndarray:
        ctx.save_for_backprop(a, b)
        return a * b
    
    def backward(self, ctx, partial: ndarray) -> Tuple[ndarray]:
        a, b = ctx.backprop_assets
        return b * partial, a * partial

class Pow(Function):

    def forward(self, ctx, base: ndarray, exp: ndarray) -> ndarray:
        ctx.save_for_backprop(base, exp)
        return base ** exp
    
    def backward(self, ctx, partial: ndarray) -> Tuple[ndarray]:
        base, exp = ctx.backprop_assets
        return exp * base ** (exp - 1) * partial, base ** exp * np.log(base) * partial

class Matmul(Function):

    def forward(self, ctx, a: ndarray, b: ndarray) -> ndarray:
        ctx.save_for_backprop(a, b)
        return a @ b
    
    def backward(self, ctx, partial: ndarray) -> Tuple[ndarray]:
        a, b = ctx.backprop_assets
        print(a.shape, b.shape, partial.shape)
        return partial @ b.swapaxes(-2, -1), a.swapaxes(-2, -1) @ partial
    
class Slice(Function):

    def forward(self, ctx, data: ndarray, slices: Any) -> ndarray:
        ctx.save_for_backprop(slices, data.shape)
        return data[slices]
    
    def backward(self, ctx, partial: ndarray) -> Tuple[ndarray]:
        slices, original_shape = ctx.backprop_assets
        template = np.zeros(original_shape)
        template[slices] += partial
        return (template, )

class Compare(Function):

    def __init__(self, compare_op: Callable) -> None:
        super().__init__()
        self.op = compare_op

    def forward(self, ctx, a: ndarray, b: ndarray | Any) -> ndarray:
        return self.op(a, b)
    
    @property
    def is_differentiable(self) -> bool:
        return False
    
class Invert(Function):

    def forward(self, ctx, a: ndarray) -> ndarray:
        return ~a
    
    @property
    def is_differentiable(self) -> bool:
        return False
    
# Useful numpy functions

class Reshape(Function):

    def forward(self, ctx, data: ndarray, new_shape: Tuple[int]) -> ndarray:
        ctx.save_for_backprop(data.shape)
        return data.reshape(new_shape)
    
    def backward(self, ctx, partial: ndarray) -> Tuple[ndarray]:
        (old_shape, ) = ctx.backprop_assets
        return (partial.reshape(old_shape), )
    
class Sum(Function):
    
    def forward(self, ctx, data: ndarray, axis: int| tuple | None, keepdims: bool) -> ndarray:
        if axis is None:
            axis = tuple(range(data.ndim))
        ctx.save_for_backprop(axis, keepdims)
        return np.sum(data, axis=axis, keepdims=keepdims)
    
    def backward(self, ctx, partial: ndarray) -> Tuple[ndarray]:
        axis, keepdims = ctx.backprop_assets
        if keepdims:
            return (partial, )
        return (np.expand_dims(partial, axis=axis), )

class Transpose(Function):
    
    def forward(self, ctx, data: ndarray, axes: List[int] | Tuple[int] | None) -> ndarray:
        if axes is None:
            axes = tuple(range(data.ndim))
        ctx.save_for_backprop(np.argsort(axes))
        return data.transpose(axes)
    
    def backward(self, ctx, partial: ndarray) -> Tuple[ndarray]:
        (axes_inv, ) = ctx.backprop_assets
        return (partial.transpose(axes_inv), )

# other elementary functions
# exp already possible with np.e **
# add log, sin, cos, tan, tanh, etc.

class Log(Function):
    # TODO
    def forward(self, ctx, *inputs: ndarray | Any) -> ndarray:
        return super().forward(*inputs)
    
    def backward(self, ctx, *partials: ndarray) -> Tuple[ndarray]:
        return super().backward(*partials)
