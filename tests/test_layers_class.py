from simflow.layers.layer_class import Layer

import numpy as np
from unittest import TestCase


class HelperClass(Layer):
    def __init__(self, config, params):
        self.config = config
        self.params = params

    def forward(self, x, train):
        pass

    def backward(self):
        """dummy
        """


class TestLayerClass(TestCase):
    def setUp(self):
        self.param = np.random.random((5, 6))
        self.helper = HelperClass("config", self.param)

    def test_misc(self):
        # not really tests, for test coverage only
        assert all(self.helper.get_params().ravel() == self.param.ravel())
        self.assertEqual(self.helper.get_config(), {})
        new_params = np.random.random((5, 6))
        self.helper.set_config(["hello", new_params])
        assert all(self.helper.get_params().ravel() == new_params.ravel())
        self.assertEqual(self.helper.config, "hello")

        self.assertEqual(self.helper.__repr__(), "Layer")

        self.helper.l_name = "test"

        self.assertEqual(self.helper.__repr__(), 'test layer')

        self.helper.set_params(np.random.random((5, 6)))
        self.helper(1, 1)
        self.helper.save_layer()

    def test_construction(self):
        with self.assertRaises(TypeError):
            Layer()

    def test_init(self):
        # test initializers
        w1 = np.random.random((5, 6))
        exp_res_xavier = np.sqrt(2.0/sum(w1.shape))
        exp_res_he = np.sqrt(2.0/w1.shape[0])
        res_xavier = Layer._initializer_(w1, 'Xavier')
        res_he = Layer._initializer_(w1, 'He')
        self.assertEqual(res_xavier, exp_res_xavier)
        self.assertEqual(res_he, exp_res_he)

        with self.assertRaises(NotImplementedError):
            Layer._initializer_(w1, "randomstring")

        w2 = np.random.random((5, 6, 7, 3))
        exp_res = np.sqrt(2.0/np.product(w2.shape[1:]))
        res_ = Layer._initializer_(w2, 'Xavier')

        self.assertEqual(exp_res, res_)

        with self.assertRaises(NotImplementedError):
            Layer._initializer_(np.random.random((3, 4, 5)), 'Xavier')
