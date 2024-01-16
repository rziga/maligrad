import numpy as np
from maligrad.nn.functional import avgpool, Variable

def test_same_ndim():
    img = Variable(np.arange(16.).reshape(4, 4))
    response = avgpool(img, (3, 3), 1, 1)

    assert np.allclose(response.data, np.array([[5, 6], [9, 10]]))

def test_smaller_kernel():
    img = Variable(np.arange(12.).reshape(3, 4))
    response = avgpool(img, (3, ), 1, 1)

    assert np.allclose(response.data, np.array([[1, 2], [5, 6], [9, 10]]))

if __name__ == "__main__":
    test_same_ndim()
    test_smaller_kernel()