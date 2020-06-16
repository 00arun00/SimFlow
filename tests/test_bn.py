import numpy as np
from simflow.layers.layers import BN
from simflow.utils.grad_check_utils import numerical_gradient_array
import unittest


class TestBatchNorm(unittest.TestCase):
    def test_bn_back_flat(self):
        eps = 1e-7
        dim = 128
        batch_size = 32

        random_inp = np.random.randn(batch_size, dim)
        beta = np.random.randn(1, dim)
        gamma = np.random.randn(1, dim)
        random_output_gradient = np.random.randn(batch_size, dim)
        learned_mean = np.random.randn(1, dim)
        learned_var = np.random.randn(1, dim)

        bn_layer = BN(dim)
        bn_layer.beta = beta
        bn_layer.gamma = gamma
        bn_layer.learned_mean = learned_mean
        bn_layer.learned_var = learned_var

        dx_num = numerical_gradient_array(lambda x: bn_layer.forward(
            random_inp), random_inp, random_output_gradient, h=eps)
        dbeta_num = numerical_gradient_array(lambda beta: bn_layer.forward(
            random_inp), beta, random_output_gradient, h=eps)
        dgamma_num = numerical_gradient_array(lambda gamma: bn_layer.forward(
            random_inp), gamma, random_output_gradient, h=eps)

        dx, grads = bn_layer.backward(random_output_gradient)
        dbeta = grads[1][1]
        dgamma = grads[0][1]

        assert np.allclose(dbeta, dbeta_num, atol=eps)
        assert np.allclose(dgamma, dgamma_num, atol=eps)
        assert np.allclose(dx, dx_num, atol=eps)

    def test_bn_back_3D(self):
        eps = 1e-7
        dim = (4, 5, 9)
        batch_size = 7

        random_inp = np.random.randn(batch_size, *dim)
        beta = np.random.randn(1, int(np.prod(dim)))
        gamma = np.random.randn(1, int(np.prod(dim)))
        random_output_gradient = np.random.randn(batch_size, *dim)
        learned_mean = np.random.randn(1, int(np.prod(dim)))

        bn_layer = BN(dim)
        bn_layer.beta = beta
        bn_layer.learned_mean = learned_mean
        bn_layer.gamma = gamma

        dx_num = numerical_gradient_array(lambda x: bn_layer.forward(
            random_inp), random_inp, random_output_gradient, h=eps)
        dbeta_num = numerical_gradient_array(lambda beta: bn_layer.forward(
            random_inp), beta, random_output_gradient, h=eps)
        dgamma_num = numerical_gradient_array(lambda gamma: bn_layer.forward(
            random_inp), gamma, random_output_gradient, h=eps)

        dx, grads = bn_layer.backward(random_output_gradient)
        dbeta = grads[1][1]
        dgamma = grads[0][1]

        assert np.allclose(dbeta, dbeta_num, atol=eps)
        assert np.allclose(dgamma, dgamma_num, atol=eps)
        assert np.allclose(dx, dx_num, atol=eps)
