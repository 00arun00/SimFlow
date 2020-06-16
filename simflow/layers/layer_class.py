import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    '''
    Abstract class representing a neural network layer
    '''

    @abstractmethod
    def forward(self, X, train=True):
        '''
        Calculates a forward pass through the layer.

        Args:
            X (numpy.ndarray): Input  (batch_size, input_size)
            train (bool): If true caches values required for backward function

        Returns:
            Out (numpy.ndarray): Output (batch_size, output_size)
        '''

    @abstractmethod
    def backward(self, dY):
        '''
        Calculates a backward pass through the layer.

        Args:
            dY (numpy.ndarray): Output grad (batch_size, output_size)

        Returns:
            dX (numpy.ndarray): Gradient of the input (batch_size, output_size)
            var_grad_list (List[Tuple[np.array]]): variables and grads
        '''

    @staticmethod
    def _initializer_(W, init_method):
        """

        Initializes the parameter passes as argument using
        Xavier or He initialization

        Args:
            W (numpy.ndarray): Parameter to be initialized
            init_method (str): Method to initialize the parameter

        """
        init_method = init_method.lower()
        if init_method not in ['xavier', 'he']:
            raise NotImplementedError('method not currently supported')

        if len(W.shape) == 2:  # linear layer
            input_dim, output_dim = W.shape
            if init_method == 'xavier':
                return np.sqrt(2.0/(input_dim+output_dim))
            if init_method == 'he':
                return np.sqrt(2.0/(input_dim))
        if len(W.shape) == 4:  # convolutional layer
            n_filter, d_filter, h_filter, w_filter = W.shape
            return np.sqrt(2.0/(h_filter*w_filter*d_filter))

        raise NotImplementedError('This W size is not defined')

    def get_params(self):
        '''
        Returns the list of numpy array of weights
        '''
        return self.params

    def set_params(self, params):
        '''
        Sets the params of a layer with a given list of numpy arrays

        Ags:
            :params (list of numpy.ndarray): new weights

        '''
        old_params = self.get_params()
        assert len(old_params) == len(params), "length mismatch"
        assert all(params[i].shape == old_params[i].shape for i in range(
            len(old_params))), "shape missmatch"
        self.params = params.copy()

    def set_config(self, config):
        self.__init__(*config)

    def save_layer(self):
        return (('conf:', self.get_config()), ('params:', self.get_params()))

    def get_config(self):
        return {}

    def __repr__(self):
        if hasattr(self, 'l_name'):
            return f'{self.l_name} layer'
        return f'Layer'

    def __call__(self, X, train=True):
        return self.forward(X, train)
