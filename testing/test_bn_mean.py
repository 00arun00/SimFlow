import numpy as np
from layers import BN_mean
from grad_check_utils import numerical_gradient_array

def test_bn_back():
    eps = 1e-7
    dim = 128
    batch_size = 32

    random_inp = np.random.randn(batch_size,dim)
    beta = np.random.randn(1,dim)
    random_output_gradient = np.random.randn(batch_size,dim)
    learned_mean = np.random.randn(1,dim)

    b_layer=BN_mean(dim)
    b_layer.beta = beta
    b_layer.learned_mean = learned_mean

    dx_num = numerical_gradient_array(lambda x: b_layer.forward(random_inp), random_inp, random_output_gradient,h=eps)
    dbeta_num = numerical_gradient_array(lambda beta: b_layer.forward(random_inp), beta, random_output_gradient,h=eps)

    out = b_layer.forward(random_inp)
    dx,grads = b_layer.backward(random_output_gradient)
    dbeta = grads[0][1]

    print("Testing backward pass of batch norm Layer")
    assert np.allclose(dx,dx_num,atol=eps)
    assert np.allclose(dbeta,dbeta_num,atol=eps)
