import numpy as np
class Layer(object):
    '''
    Abstract class representing a neural network layer
    '''
    def forward(self, X, train=True):
        '''
        Calculates a forward pass through the layer.

        Args:
            :X (numpy.ndarray):   Input to the layer with dimensions (batch_size, input_size)
            :train (bool):   If true caches values required for backward function

        Returns:
            :Out (numpy.ndarray):   Output of the layer with dimensions (batch_size, output_size)
        '''
        raise NotImplementedError('This is an abstract class')

    def backward(self, dY):
        '''
        Calculates a backward pass through the layer.

        Args:
            :dY (numpy.ndarray):   The gradient of the output with dimensions (batch_size, output_size)

        Returns:
            :dX (numpy.ndarray):   Gradient of the input (batch_size, output_size)
            :var_grad_list (list):   List of tuples in the form (variable_pointer, variable_grad)
        '''
        raise NotImplementedError('This is an abstract class')

    def _initializer_(self,W,init_method):
        """
        Initializes the parameter passes as argument using Xavier of He initialization

        Args:
            W (numpy.ndarray): Parameter to be initialized
            init_method (str): Method to initialize the parameter

        """
        if init_method == 'Xavier':
            if len(W.shape)==2: #linear layer
              input_dim,output_dim = W.shape
              return np.sqrt(2.0/(input_dim+output_dim))
            elif len(W.shape)==4:#convolutional layer
              n_filter,d_filter,h_filter,w_filter = W.shape
              return np.sqrt(2.0/(h_filter*w_filter*d_filter))
            else:
              raise NotImplementedError('This W size is not defined')
        elif init_method == 'He':
            if len(W.shape)==2: #linear layer
              input_dim,output_dim = W.shape
              return np.sqrt(2.0/(input_dim))
            elif len(W.shape)==4:#convolutional layer
              n_filter,d_filter,h_filter,w_filter = W.shape
              return np.sqrt(2.0/(h_filter*w_filter*d_filter))
            else:
              raise NotImplementedError('This W size is not defined')
        else:
          raise NotImplementedError('This method not currently supported')

    def get_params(self):
        '''
        Returns the list of numpy array of weights
        '''
        if hasattr(self,'params'):
            return self.params
        else:
            raise ValueError('Params not defined')

    def set_params(self,params):
        '''
        Sets the params of a layer with a given list of numpy arrays

        Ags:
            :params (list of numpy.ndarray): new weights

        '''
        old_params = self.get_params()
        assert len(old_params) == len(params),"length mismatch"
        assert all(params[i].shape == old_params[i].shape for i in range(len(old_params))),"shape missmatch"
        self.params = params.copy()

    def set_config(self,config):
        self.__init__(self,*config)

    def save_layer(self):
        return [('conf:',self.get_config()),('params:',self.get_params())]

    def _get_config_(self):
        return {}

    def __repr__(self):
        if hasattr(self,'l_name'):
            return f'{self.l_name} layer'
        else:
            return f'Layer'

    def __call__(self,X,train=True):
        return self.forward(X,train)
