import numpy as np
from simflow.layers.convolutional import dilated_Conv2D
from simflow.utils.grad_check_utils import numerical_gradient_array
import unittest


class TestDilatedConv(unittest.TestCase):
    def test_dilated_conv_layer_back_prop(self):
        eps = 1e-7
        batch_size = 32
        filter_size = 3
        h_x, w_x = 7, 7
        inChannels = 3
        n_filter = 5
        padding = 0
        stride = 1
        dilation = 2
        new_filter_size = dilation*(filter_size-1)+1
        h_out = (h_x - new_filter_size + 2*padding)//stride + 1
        w_out = (w_x - new_filter_size + 2*padding)//stride + 1

        x = np.random.randn(batch_size, inChannels, h_x, w_x)
        w = np.random.randn(n_filter, inChannels, filter_size, filter_size)
        b = np.random.randn(n_filter, 1)
        dout = np.random.randn(batch_size, n_filter, h_out, w_out)

        dc_layer = dilated_Conv2D(inChannels=inChannels, outChannels=n_filter,
                                  filter_size=filter_size, stride=stride,
                                  padding=padding, trainable=True)
        dc_layer.W = w
        dc_layer.b = b

        dx_num = numerical_gradient_array(dc_layer.forward,
                                          x, dout, h=eps)
        dw_num = numerical_gradient_array(lambda w: dc_layer.forward(x),
                                          w, dout, h=eps)
        db_num = numerical_gradient_array(lambda b: dc_layer.forward(x),
                                          b, dout, h=eps)

        dx, grads = dc_layer.backward(dout)
        dw, db = grads[0][1], grads[1][1]

        assert np.allclose(dx, dx_num, atol=eps)
        assert np.allclose(dw, dw_num, atol=eps)
        assert np.allclose(db, db_num, atol=eps)
