# source https://github.com/parasdahal/deepnet
import numpy as np


def numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.

    Args:
        f (function): function that is passed to compute gradient for
        x (numpy.ndarray): input to function
        df (numpy.ndarray): output gradient
        h (float): delta around which gradient is calculated

    Returns:
        grad (numpy.ndarray): computed numeric gradient
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)

        it.iternext()
    return grad
