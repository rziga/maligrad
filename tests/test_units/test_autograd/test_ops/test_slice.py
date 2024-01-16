import numpy as np

from maligrad.autograd.engine import Variable, Slice

from .base import OpTester


def test_single():
    tester = OpTester(
        fcn=Slice(),
        inputs=[
            Variable(np.ones((3, 3)), True), 
            (1, 1)
            ], 
        expected_output=np.array(1), 
        backward_seed=np.array(1),
        expected_grads=[
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), 
            None,
            ]
        )
    tester.test_backward()
    tester.test_forward()

def test_slice():
    tester = OpTester(
        fcn=Slice(),
        inputs=[
            Variable(np.ones((3, 3)), True), 
            (slice(None, None, None), 1) # slice(None, None, None) <=> :
            ], 
        expected_output=np.ones(3),
        backward_seed=np.ones(3),
        expected_grads=[
            np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]), 
            None,
            ]
        )
    tester.test_backward()
    tester.test_forward()

def test_bool_indexing():
    tester = OpTester(
        fcn=Slice(),
        inputs=[
            Variable(np.ones((3, 3)), True), 
            Variable(np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool))
            ], 
        expected_output=np.ones(3),
        backward_seed=np.ones(3),
        expected_grads=[
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]), 
            None,
            ]
        )
    tester.test_backward()
    tester.test_forward()

def test_condition_indexing():
    x = Variable(np.arange(9.).reshape(3, 3), True)
    inds = x > 5 # [[0, 0, 0], [0, 0, 0], [1, 1, 1]]
    tester = OpTester(
        fcn=Slice(),
        inputs=[
            x,
            inds
            ], 
        expected_output=np.array([6, 7, 8]),
        backward_seed=np.ones(3),
        expected_grads=[
            np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]), 
            None,
            ]
        )
    tester.test_backward()
    tester.test_forward()

def test_range_indexing():
    tester = OpTester(
        fcn=Slice(),
        inputs=[
            Variable(np.arange(9.).reshape(3, 3), True),
            (range(3), range(3)[::-1])
            ], 
        expected_output=np.array([2, 4, 6]),
        backward_seed=np.ones(3),
        expected_grads=[
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]), 
            None,
            ]
        )
    tester.test_backward()
    tester.test_forward()

def test_array_indexing():
    tester = OpTester(
        fcn=Slice(),
        inputs=[
            Variable(np.arange(9.).reshape(3, 3), True),
            ([1, 2, 2, 0], [2, 1, 0, 1])
            ], 
        expected_output=np.array([5, 7, 6, 1]),
        backward_seed=np.ones(4),
        expected_grads=[
            np.array([[0, 1, 0], [0, 0, 1], [1, 1, 0]]), 
            None,
            ]
        )
    tester.test_backward()
    tester.test_forward()

def test_array_indexing_overlap():
    tester = OpTester(
        fcn=Slice(),
        inputs=[
            Variable(np.arange(3.), True),
            ([0, 1, 1])
            ], 
        expected_output=np.array([0, 1, 1]),
        backward_seed=np.ones(3),
        expected_grads=[
            np.array([1, 2, 0]), 
            None,
            ]
        )
    tester.test_backward()
    tester.test_forward()

if __name__ == "__main__":
    test_single()
    test_slice()
    test_bool_indexing()
    test_condition_indexing()
    test_range_indexing()
    test_array_indexing()
