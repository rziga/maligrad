import numpy as np
from maligrad.nn.functional import maxpool, Variable

def test_same_ndim():
    img = Variable(np.arange(16.).reshape(4, 4))
    response = maxpool(img, (3, 3), 1, 1)

    assert np.allclose(response.data, np.array([[10, 11], [14, 15]]))

def test_smaller_kernel():
    img = Variable(np.arange(12.).reshape(3, 4))
    response = maxpool(img, (3, ), 1, 1)

    assert np.allclose(response.data, np.array([[2, 3], [6, 7], [10, 11]]))

if __name__ == "__main__":
    test_same_ndim()
    test_smaller_kernel()
