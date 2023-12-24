import operator
import numpy as np
import pytest

from milligrad.autograd.ops import Compare
from milligrad.autograd.engine import DataNode

from .base import OpTester


def test_basic():
    tester = OpTester(
        fcn=Compare(operator.eq),
        inputs=[
            DataNode(np.arange(9).reshape(3, 3), True),
            DataNode(np.arange(9)[::-1].reshape(3, 3), True)
            ], 
        expected_output=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), 
        backward_seed=None, 
        expected_grads=[
            None, 
            None,
            ]
        )
    with pytest.raises(AssertionError):
        tester.test_backward()
    tester.test_forward()

def test_broadcast():
    tester = OpTester(
        fcn=Compare(operator.gt),
        inputs=[
            DataNode(np.arange(4).reshape(2, 2), True),
            DataNode(1, True),
            ],
        expected_output=np.array([[0, 0], [1, 1]], dtype=bool),
        backward_seed=None,
        expected_grads=[
            None,
            None,
            ]
        )
    with pytest.raises(AssertionError):
        tester.test_backward()
    tester.test_forward()


if __name__ == "__main__":
    test_basic()
    test_broadcast()
