import numpy as np

from autograd.ops import Sum
from autograd.engine import DataNode

from tests.test_ops.base import OpTester

def test_basic():
    tester = OpTester(
        fcn=Sum(),
        inputs=[
            DataNode(np.array([1, 2, 3, 4]), True),
            None,
            False
            ], 
        expected_output=np.array(10), 
        backward_seed=np.array(1),
        expected_grads=[
            np.array([1, 1, 1, 1]),
            None,
            None,
            ]
        )
    tester.test_backward()
    tester.test_forward()

def test_single_axis():
    tester = OpTester(
        fcn=Sum(),
        inputs=[
            DataNode(np.array([[1., 2], [3, 4]]), True),
            0,
            False
            ], 
        expected_output=np.array([4., 6]), 
        backward_seed=np.array([1., 2]),
        expected_grads=[
            np.array([[1., 2], [1, 2]]),
            None,
            None,
            ]
        )
    tester.test_backward()
    tester.test_forward()

def test_keepdims():
    tester = OpTester(
        fcn=Sum(),
        inputs=[
            DataNode(np.array([[1., 2], [3, 4]]), True),
            None,
            True
            ], 
        expected_output=np.array([[10.]]), 
        backward_seed=np.array([[1.]]),
        expected_grads=[
            np.array([[1., 1], [1, 1]]),
            None,
            None,
            ]
        )
    tester.test_backward()
    tester.test_forward()

if __name__ == "__main__":
    test_basic()
    test_single_axis()
    test_keepdims()