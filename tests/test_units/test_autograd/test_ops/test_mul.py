import numpy as np

from milligrad.autograd.ops import Mul
from milligrad.autograd.engine import DataNode

from .base import OpTester


def test_basic():
    shape = (3, 3)
    tester = OpTester(
        fcn=Mul(), 
        inputs=[
            DataNode(np.ones(shape), True),
            DataNode(2*np.ones(shape), True),
            ],
        expected_output=2*np.ones(shape),
        backward_seed=np.ones(shape),
        expected_grads=[
            2*np.ones(shape),
            np.ones(shape),
            ],
        )
    tester.test_backward()
    tester.test_forward()

def test_broadcast():
    tester = OpTester(
        fcn=Mul(),
        inputs=[
            DataNode(2*np.ones((2, 1, 2)), True),
            DataNode(3*np.ones((2, 1)), True),
            ],
        expected_output=2*3*np.ones((2, 2, 2)),
        backward_seed=np.ones((2, 2, 2)),
        expected_grads=[
            2*3*np.ones((2, 1, 2)),
            4*2*np.ones((2, 1)),
            ]
        )
    tester.test_backward()
    tester.test_forward()

if __name__ == "__main__":
    test_basic()
    test_broadcast()