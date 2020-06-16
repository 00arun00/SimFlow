import numpy as np
from simflow.layers.activations import (
    ReLU, LeakyReLU, Softplus, exp, Sigmoid, Tanh
)
from simflow.utils.grad_check_utils import numerical_gradient_array
import unittest


class TestActivations(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-7
        self.batch_size = 32
        self.h_x, self.w_x = 7, 7
        self.inChannels = 3

        self.x = np.random.randn(self.batch_size, self.inChannels,
                                 self.h_x, self.w_x)
        self.dout = np.random.randn(self.batch_size, self.inChannels,
                                    self.h_x, self.w_x)

    def helper_grad_check(self, Layer):
        layer = Layer(trainable=True)
        dx_num = numerical_gradient_array(lambda x: layer.forward(self.x),
                                          self.x, self.dout, h=self.eps)
        dx, _ = layer.backward(self.dout)

        assert np.allclose(dx, dx_num, atol=self.eps)

        with self.assertRaises(RuntimeError):
            layer2 = Layer(trainable=False)
            layer2.backward(self.dout)

    def test_relu_back_prop(self):
        self.helper_grad_check(ReLU)

    def test_LeakyReLU_back_prop(self):
        self.helper_grad_check(LeakyReLU)

    def test_Softplus_back_prop(self):
        self.helper_grad_check(Softplus)

    def test_exp_back_prop(self):
        self.helper_grad_check(exp)

    def test_sigmoid_back_prop(self):
        self.helper_grad_check(Sigmoid)

    def test_tanh_back_prop(self):
        self.helper_grad_check(Tanh)
