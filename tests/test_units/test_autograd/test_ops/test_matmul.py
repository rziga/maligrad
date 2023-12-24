import numpy as np

from milligrad.autograd.ops import Matmul
from milligrad.autograd.engine import DataNode

from .base import OpTester


def test_basic():
    shape1 = (2, 4)
    shape2 = (4, 3)
    shape_out = (shape1[0], shape2[1])
    a = DataNode(2*np.ones(shape1), True)
    b = DataNode(3*np.ones(shape2), True)

    tester = OpTester(
        fcn=Matmul(),
        inputs=[a, b],
        expected_output=shape1[1] * 2 * 3 * np.ones(shape_out),
        backward_seed=np.ones(shape_out),
        expected_grads=[
            np.ones(shape_out) @ b.data.T, 
            a.data.T @ np.ones(shape_out),
            ],
        )
    tester.test_backward()
    tester.test_forward()

def test_broadcast():
    shape1 = (2, 1, 2, 3, 4)
    shape2 = (2, 1, 4, 1)
    shape_out = (2, 2, 2, 3, 1)
    a = DataNode(2*np.ones(shape1), True)
    b = DataNode(3*np.ones(shape2), True)

    tester = OpTester(
        fcn=Matmul(),
        inputs=[a, b],
        expected_output= 4 * 2 * 3 * np.ones(shape_out),
        backward_seed=np.ones(shape_out),
        expected_grads=[
            2 * np.ones(shape_out[-2:]) @ b.data.swapaxes(-1, -2),
            4 * a.data.swapaxes(-1, -2) @ np.ones(shape_out[-2:]),
            ]
        )
    tester.test_backward()
    tester.test_forward()

if __name__ == "__main__":
    test_basic()
    test_broadcast()