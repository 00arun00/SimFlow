import numpy as np
from simflow.layers.layers import Dense
from simflow.utils.grad_check_utils import numerical_gradient_array
import unittest


class TestDense(unittest.TestCase):
    @staticmethod
    def standard_forward_Dense(X, w, b):
        return X@w + b

    def test_Dense_forward(self):
        batch_size = 32
        inp_dim = 10
        out_dim = 5
        eps = 1e-7
        Input = np.random.randn(batch_size, inp_dim)

        d_layer = Dense(inp_dim, out_dim, init_method='He')

        W = np.random.randn(inp_dim, out_dim)
        b = np.random.randn(1, out_dim)

        d_layer.W = W
        d_layer.b = b
        assert np.allclose(d_layer.forward(Input), self.standard_forward_Dense(
            Input, W, b), atol=eps), "forward not working"

    def test_linear_back(self):
        eps = 1e-7
        input_dim = 10
        output_dim = 5

        batch_size = 32

        random_inp = np.random.randn(batch_size, input_dim)
        W = np.random.randn(input_dim, output_dim)
        b = np.random.randn(1, output_dim)
        random_output_gradient = np.random.randn(batch_size, output_dim)

        lin_layer = Dense(input_dim, output_dim, init_method='He')
        lin_layer.W = W
        lin_layer.b = b

        dx_num = numerical_gradient_array(lambda x: lin_layer.forward(
            random_inp), random_inp, random_output_gradient, h=eps)
        dw_num = numerical_gradient_array(lambda W: lin_layer.forward(
            random_inp), W, random_output_gradient, h=eps)
        db_num = numerical_gradient_array(lambda b: lin_layer.forward(
            random_inp), b, random_output_gradient, h=eps)

        dx, grads = lin_layer.backward(random_output_gradient)
        dW, db = grads[0][1], grads[1][1]

        assert np.allclose(dx, dx_num, atol=eps)
        assert np.allclose(dW, dw_num, atol=eps)
        assert np.allclose(db, db_num, atol=eps)
