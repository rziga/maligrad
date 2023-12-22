import numpy as np

from autograd.ops import Transpose
from autograd.engine import DataNode

from tests.test_ops.base import OpTester

def test_basic():
    tester = OpTester(
        fcn=Transpose(),
        inputs=[
            DataNode(np.array([[1, 2], [3, 4]]), True),
            (1, 0)
            ], 
        expected_output=np.array([[1, 3], [2, 4]]), 
        backward_seed=np.array([[1, 2], [1, 2]]),
        expected_grads=[
            np.array([[1, 1], [2, 2]]),
            None,
            ]
        )
    tester.test_backward()
    tester.test_forward()

def test_auto_axis():
    tester = OpTester(
        fcn=Transpose(),
        inputs=[
            DataNode(np.array([[1, 2], [3, 4]]), True),
            None
            ], 
        expected_output=np.array([[1, 3], [2, 4]]), 
        backward_seed=np.array([[1, 2], [1, 2]]),
        expected_grads=[
            np.array([[1, 1], [2, 2]]),
            None,
            ]
        )
    tester.test_backward()
    tester.test_forward()

if __name__ == "__main__":
    test_basic()
    test_auto_axis()