import numpy as np

from maligrad.autograd.ops import Add
from maligrad.autograd.engine import DataNode

from .base import OpTester


def test_basic():
    shape = (3, 3)
    tester = OpTester(
        fcn=Add(),
        inputs=[
            DataNode(np.ones(shape), True), 
            DataNode(2*np.ones(shape), True)
            ], 
        expected_output=3*np.ones(shape), 
        backward_seed=np.ones(shape), 
        expected_grads=[
            np.ones(shape), 
            np.ones(shape),
            ]
        )
    tester.test_backward()
    tester.test_forward()

def test_broadcast():
    tester = OpTester(
        fcn=Add(),
        inputs=[
            DataNode(np.ones((2, 1, 2)), True),
            DataNode(2*np.ones((2, 1)), True),
            ],
        expected_output=3*np.ones((2, 2, 2)),
        backward_seed=np.ones((2, 2, 2)),
        expected_grads=[
            2*np.ones((2, 1, 2)),
            4*np.ones((2, 1))
            ]
        )
    tester.test_backward()
    tester.test_forward()


if __name__ == "__main__":
    test_basic()
    test_broadcast()
