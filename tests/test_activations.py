import numpy as np
from layers import ReLU,LeakyReLU,Softplus,exp
from grad_check_utils import numerical_gradient_array

def test_relu_back_prop():
    eps = 1e-7

    batch_size = 32
    h_x,w_x = 7,7
    inChannels = 3

    x = np.random.randn(batch_size, inChannels, h_x, w_x)
    dout = np.random.randn(batch_size, inChannels, h_x, w_x)

    r_layer = ReLU(trainable=True)

    dx_num = numerical_gradient_array(lambda x: r_layer.forward(x), x, dout,h=eps)


    out = r_layer(x)
    dx,_ = r_layer.backward(dout)

    assert np.allclose(dx,dx_num,atol=eps)

def test_LeakyReLU_back_prop():
    eps = 1e-7

    batch_size = 32
    h_x,w_x = 7,7
    inChannels = 3

    x = np.random.randn(batch_size, inChannels, h_x, w_x)
    dout = np.random.randn(batch_size, inChannels, h_x, w_x)

    a_layer = LeakyReLU(alpha=0.1,trainable=True)

    dx_num = numerical_gradient_array(lambda x: a_layer.forward(x), x, dout,h=eps)


    out = a_layer(x)
    dx,_ = a_layer.backward(dout)

    assert np.allclose(dx,dx_num,atol=eps)


def test_Softplus_back_prop():
    eps = 1e-7

    batch_size = 32
    h_x,w_x = 7,7
    inChannels = 3

    x = np.random.randn(batch_size, inChannels, h_x, w_x)
    dout = np.random.randn(batch_size, inChannels, h_x, w_x)

    a_layer = Softplus(trainable=True)

    dx_num = numerical_gradient_array(lambda x: a_layer.forward(x), x, dout,h=eps)


    out = a_layer(x)
    dx,_ = a_layer.backward(dout)

    assert np.allclose(dx,dx_num,atol=eps)

def test_exp_back_prop():
    eps = 1e-7

    batch_size = 32
    h_x,w_x = 7,7
    inChannels = 3

    x = np.random.randn(batch_size, inChannels, h_x, w_x)
    dout = np.random.randn(batch_size, inChannels, h_x, w_x)

    a_layer = exp(trainable=True)

    dx_num = numerical_gradient_array(lambda x: a_layer.forward(x), x, dout,h=eps)


    out = a_layer(x)
    dx,_ = a_layer.backward(dout)

    assert np.allclose(dx,dx_num,atol=eps)
