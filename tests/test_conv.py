import numpy as np
from simflow.layers.convolutional import Conv2D
from simflow.utils.grad_check_utils import numerical_gradient_array
from simflow.utils.im2col import im2col_indices
from unittest import TestCase


class TestConv(TestCase):
    @staticmethod
    def standard_forward(X, w, b, padding, stride):
        n_x, d_x, h_x, w_x = X.shape
        n_filters, d_filter, h_filter, w_filter = w.shape
        h_out = (h_x - h_filter + 2 * padding) // stride + 1
        w_out = (w_x - w_filter + 2 * padding) // stride + 1
        X_col = im2col_indices(X, h_filter, w_filter,
                               padding=padding, stride=stride)
        W_col = w.reshape(n_filters, -1)
        out = W_col @ X_col + b
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        return out

    def test_conv_layer_forward_prop(self):
        eps = 1e-8

        batch_size = 32
        filter_size = 3
        h_x, w_x = 7, 7
        inChannels = 3
        n_filter = 5
        padding = 1
        stride = 1

        x = np.random.randn(batch_size, inChannels, h_x, w_x)
        w = np.random.randn(n_filter, inChannels, filter_size, filter_size)
        b = np.random.randn(n_filter, 1)

        c_layer = Conv2D(inChannels=inChannels, outChannels=n_filter,
                         filter_size=filter_size, stride=stride,
                         padding=padding, trainable=True)
        c_layer.W = w
        c_layer.b = b

        out = c_layer.forward(x)
        out_standard = self.standard_forward(x, w, b, padding, stride)

        assert np.allclose(out, out_standard, atol=eps)

    def test_conv_layer_back_prop(self):
        eps = 1e-7

        batch_size = 32
        filter_size = 3
        h_x, w_x = 7, 7
        inChannels = 3
        n_filter = 5
        padding = 1
        stride = 1

        h_out = (h_x - filter_size + 2*padding)//stride + 1
        w_out = (w_x - filter_size + 2*padding)//stride + 1

        x = np.random.randn(batch_size, inChannels, h_x, w_x)
        w = np.random.randn(n_filter, inChannels, filter_size, filter_size)
        b = np.random.randn(n_filter, 1)
        dout = np.random.randn(batch_size, n_filter, h_out, w_out)

        c_layer = Conv2D(inChannels=inChannels, outChannels=n_filter,
                         filter_size=filter_size, stride=stride,
                         padding=padding, trainable=True)
        c_layer.W = w
        c_layer.b = b

        dx_num = numerical_gradient_array(c_layer.forward, x, dout, h=eps)
        dw_num = numerical_gradient_array(lambda w: c_layer.forward(x),
                                          w, dout, h=eps)
        db_num = numerical_gradient_array(lambda b: c_layer.forward(x),
                                          b, dout, h=eps)

        dx, grads = c_layer.backward(dout)
        dw, db = grads[0][1], grads[1][1]

        assert np.allclose(dx, dx_num, atol=eps)
        assert np.allclose(dw, dw_num, atol=eps)
        assert np.allclose(db, db_num, atol=eps)
