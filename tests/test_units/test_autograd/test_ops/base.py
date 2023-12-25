from typing import List, Any
from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from maligrad.autograd.ops import Function
from maligrad.autograd.engine import FunctionNode, DataNode


@dataclass
class OpTester():
    fcn: Function
    inputs: List[DataNode | Any]
    expected_output: ndarray
    backward_seed: ndarray
    expected_grads: List[ndarray | None]

    __test__ = False
    
    def test_forward(self):
        fcn_node = FunctionNode(self.fcn)
        out = fcn_node(*self.inputs)
        assert np.allclose(out.data, self.expected_output)

    def test_backward(self):
        fcn_node = FunctionNode(self.fcn)
        out = fcn_node(*self.inputs)
        out.backward(self.backward_seed)
        for input_, expected_grad in zip(self.inputs, self.expected_grads):
            if isinstance(input_, DataNode) and input_.requires_grad:
                assert np.allclose(input_.grad, expected_grad)