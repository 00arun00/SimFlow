from simflow.layers.layer_class import Layer
import numpy as np


class ReLU(Layer):
    '''
    RelU layer
    Represent a nonlinear transformation Y = max(0,X)
    '''

    def __init__(self, *, trainable=True):
        '''
        Initialization :
            Does nothing since nothing to initialize
        '''
        self.cache_in = None
        self.trainable = trainable
        self.l_name = 'ReLU'
        self.params = []

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the ReLU Layer

        Args:
            :X (numpy.ndarray): Input array
            :train (bool): Set to True to enable gardient caching for backward step

        Returns:
            :Out (numpy.ndarray): Output after applying transformation Y = max(0,X)
        '''
        out = np.maximum(X, 0)
        if train:
            self.cache_in = X
        return out

    def backward(self, dY):
        '''
        Performs a backward pass through the ReLU Layer

        Args:
            :dY (numpy.ndarray): Output gradient backpropagated from layers in the front

        Returns:
            :dX (numpy.ndarray): Input gradient after backpropagating dY through ReLU layer
            :var_grad_list (list): [], since it has no parameter to be learned
        '''
        if self.cache_in is None:
            raise RuntimeError(
                'Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        dX = dY*(self.cache_in >= 0)
        return dX, []


# adding aliases
relu = ReLU


class Sigmoid(Layer):
    '''
    Sigmoid layer
    Represent a nonlinear transformation Y = 1/(1+e^(-X))
    '''

    def __init__(self, *, trainable=True):
        '''
        Initialization:
            Does nothing since nothing to initialize
        '''
        self.cache_in = None
        self.trainable = trainable
        self.l_name = 'Sigmoid'
        self.params = []

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the Sigmoid Layer

        Args:
            :X (numpy.ndarray): Input array
            :train (bool): Set to True to enable caching for backward step

        Returns:
            :Out (numpy.ndarray): Output after applying transformation Y = 1/(1+e^(-X))
        '''
        out = 1/(1+np.exp(-X))
        if train:
            self.cache_in = out
        return out

    def backward(self, dY):
        '''
        Performs a backward pass through the Sigmoid Layer

        Args:
            :dY (numpy.ndarray): Output gradient backpropagated from layers in the front

        Returns:
            :dX (numpy.ndarray): Input gradient after backpropagating dY through Sigmoid layer
            :var_grad_list (list): [], since it has no parameter to be learned
        '''
        if self.cache_in is None:
            raise RuntimeError(
                'Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        out = self.cache_in
        dX = dY*(out*(1-out))
        return dX, []


class Tanh(Layer):
    '''
    Tanh layer
    Represent a nonlinear transformation Y = (1-e^(-2X))/(1+e^(-2X)) {tanh}
    '''

    def __init__(self, *, trainable=True):
        '''
        Initialization :
            Does nothing since nothing to initialize
        '''
        self.cache_in = None
        self.trainable = trainable
        self.l_name = 'Tanh'
        self.params = []

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the Tanh Layer

        Args:
            :X (numpy.ndarray): Input array
            :train (bool): Set to True to enable caching for backward step

        Returns:
            :Out (numpy.ndarray): Output after applying transformation Y = tanh(X)
        '''
        out = np.tanh(X)
        if train:
            self.cache_in = out
        return out

    def backward(self, dY):
        '''
        Performs a backward pass through the Tanh Layer

        Args:
            :dY (numpy.ndarray): Output gradient backpropagated from layers in the front

        Returns:
            :dX (numpy.ndarray): Input gradient after backpropagating dY through Tanh layer
            :var_grad_list (list): [], since it has no parameter to be learned
        '''

        if self.cache_in is None:
            raise RuntimeError(
                'Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        out = self.cache_in
        dX = dY*(1-out**2)
        return dX, []


class LeakyReLU(Layer):
    '''
    LeakyReLU layer
    Represent a nonlinear transformation Y = max(alpha*input,input)

    Args:
        :alpha (float): Leakiness parameter
    '''

    def __init__(self, alpha=0.1, *, trainable=True):
        '''
        Initialization :
            initialize the leakiness parameter
        '''
        self.cache_in = None
        self.trainable = trainable
        self.l_name = 'LeakyReLU'
        self.params = []
        self.alpha = alpha

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the LeakyReLU Layer

        Args:
            :X (numpy.ndarray): Input array
            :train (bool): Set to True to enable caching for backward step

        Returns:
            :Out (numpy.ndarray): Output after applying transformation Y = LeakyReLU(X)
        '''
        out = np.maximum(self.alpha*X, X)
        if train:
            self.cache_in = X
        return out

    def backward(self, dY):
        '''
        Performs a backward pass through the LeakyReLU Layer

        Args:
            :dY (numpy.ndarray): Output gradient backpropagated from layers in the front

        Returns:
            :dX (numpy.ndarray): Input gradient after backpropagating dY through LeakyReLU layer
            :var_grad_list (list): [], since it has no parameter to be learned
        '''

        if self.cache_in is None:
            raise RuntimeError(
                'Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        X = self.cache_in
        grad = 1*(X > 0)+self.alpha*(X <= 0)
        dX = dY*grad
        return dX, []


class Softplus(Layer):
    '''
    Represent a nonlinear transformation Y = log(1+exp(X))
    '''

    def __init__(self, *, trainable=True):
        '''
        Initialization :
            Nothing to initialize
        '''
        self.cache_in = None
        self.trainable = trainable
        self.l_name = 'LeakyReLU'
        self.params = []

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the Softplus Layer

        Args:
            :X (numpy.ndarray): Input array
            :train (bool): Set to True to enable caching for backward step

        Returns:
            :Out (numpy.ndarray): Output after applying transformation Y = Softplus(X)
        '''

        out = np.log(1+np.exp(X))
        if train:
            self.cache_in = X
        return out

    def backward(self, dY):
        '''
        Performs a backward pass through the Softplus Layer

        Args:
            :dY (numpy.ndarray): Output gradient backpropagated from layers in the front

        Returns:
            :dX (numpy.ndarray): Input gradient after backpropagating dY through Softplus layer
            :var_grad_list (list): [], since it has no parameter to be learned
        '''

        if self.cache_in is None:
            raise RuntimeError(
                'Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        X = self.cache_in
        grad = 1/(1+np.exp(-X))
        dX = dY*grad
        return dX, []


class exp(Layer):
    '''
    Represent a nonlinear transformation Y = exp(X)
    '''

    def __init__(self, *, trainable=True):
        '''
        Initialization :
            Nothing to initialize
        '''
        self.cache_in = None
        self.trainable = trainable
        self.l_name = 'LeakyReLU'
        self.params = []

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the exp Layer

        Args:
            :X (numpy.ndarray): Input array
            :train (bool): Set to True to enable caching for backward step

        Returns:
            :Out (numpy.ndarray): Output after applying transformation Y = exp(X)
        '''

        out = np.exp(X)
        if train:
            self.cache_in = X
        return out

    def backward(self, dY):
        '''
        Performs a backward pass through the Softplus Layer

        Args:
            :dY (numpy.ndarray): Output gradient backpropagated from layers in the front

        Returns:
            :dX (numpy.ndarray): Input gradient after backpropagating dY through exp layer
            :var_grad_list (list): [], since it has no parameter to be learned
        '''

        if self.cache_in is None:
            raise RuntimeError(
                'Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        X = self.cache_in
        grad = np.exp(X)
        dX = dY*grad
        return dX, []
