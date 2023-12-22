import numpy as np

from autograd.ops import Reshape
from autograd.engine import DataNode

from tests.test_ops.base import OpTester

def test_basic():
    tester = OpTester(
        fcn=Reshape(),
        inputs=[
            DataNode(np.array([1, 2, 3, 4]), True),
            (2, 2)
            ], 
        expected_output=np.array([[1, 2], [3, 4]]), 
        backward_seed=np.array([[0, 1], [1, 0]]),
        expected_grads=[
            np.array([0, 1, 1, 0]),
            None,
            ]
        )
    tester.test_backward()
    tester.test_forward()

def test_auto_axis():
    tester = OpTester(
        fcn=Reshape(),
        inputs=[
            DataNode(np.array([1, 2, 3, 4]), True),
            (1, -1, 1, 2) # <=> (1, 2, 1, 2)
            ], 
        expected_output=np.array([[[1, 2]], [[3, 4]]]), 
        backward_seed=np.array([[[0, 1]], [[1, 0]]]),
        expected_grads=[
            np.array([0, 1, 1, 0]),
            None,
            ]
        )
    tester.test_backward()
    tester.test_forward()

if __name__ == "__main__":
    test_basic()
    test_auto_axis()
