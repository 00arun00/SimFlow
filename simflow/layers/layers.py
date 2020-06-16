import numpy as np
from simflow.layers.layer_class import Layer


class Dense(Layer):
    '''Dense / Linear Layer

    Represent a linear transformation Y = X*W + b
        - X: is an numpy.ndarray with shape (batch_size, input_dim)
        - W: is a trainable matrix with dimensions (input_dim, output_dim)
        - b: is a bias with dimensions (1, output_dim)
        - Y: is an numpy.ndarray with shape (batch_size, output_dim)
    '''

    def __init__(self, input_dim, output_dim, *,
                 init_method='Xavier', trainable=True):
        '''
        Initializes the Desnse layer parameter
            - W: is initialized with either Xavier or He initialization
            - b: is initialized to zero

        Args:
            input_dim (int): size of input passed
            output_dim (int): size of output requred
            init_method (str): initialization method to be used for Weights
            trainable (bool): False parameters of the layer are frozen
                              True parameters updated during optimizer step
        '''
        self.init_method = init_method
        self.W = np.random.randn(input_dim, output_dim)
        self.W *= self._initializer_(self.W, init_method)
        self.b = np.zeros((1, output_dim))
        self.cache_in = None
        self.trainable = trainable
        self.l_name = 'Dense'
        self.params = [self.W, self.b]

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the Dense Layer

        Args:
            X (numpy.ndarray): Input array (batch_size x input_dim)
            train (bool): True -> enables gardient caching for backward step

        Returns:
            Out (numpy.ndarray): Output after transformation Y = X*W + b
                                 shape of output is (batch_size x output_dim)
        '''
        assert len(X.shape) == 2, "input dimenstions not supported"
        assert X.shape[1] == self.W.shape[0], (
            f"input dim doesn't match, each X has dimension {X.shape[1]} "
            f"but Weights defined are of shape {self.W.shape[0]}"
        )
        out = X@self.W + self.b
        if train:
            self.cache_in = X
        return out

    def backward(self, dY):
        '''
        Performs a backward pass through the Dense Layer

        Args:
            dY (numpy.ndarray): Output gradient backpropagated
                                shape of dY is (batch_size x output_dim)

        Returns:
            dX (numpy.ndarray): Input gradient
            var_grad_list (list):
                trainable = True: [(W,dW),(b,db)]
                trainable = False: [ ]
        '''
        dX = dY@self.W.T
        if self.trainable:
            if self.cache_in is None:
                raise RuntimeError(
                    'Gradient cache not defined. When training the train '
                    f'argument must be set to true in the forward pass.'
                )
            X = self.cache_in
            db = np.sum(dY, axis=0, keepdims=True)
            dW = X.T@dY
            assert X.shape == dX.shape, (
                f"Dimensions of grad and variable should match, "
                f"X has shape {X.shape} and dX has shape {dX.shape}"
            )
            assert self.W.shape == dW.shape, (
                f"Dimensions of grad and variable should match, "
                f"W has shape {self.W.shape} and dW has shape {dW.shape}"
            )
            assert self.b.shape == db.shape, (
                f"Dimensions of grad and variable should match, "
                f"b has shape {self.b.shape} and db has shape {db.shape}"
            )
            return dX, [(self.W, dW), (self.b, db)]
        # if not trainable
        return dX, []

    def __repr__(self):
        return f'Dense Layer with shape {self.W.shape}'

    def _get_config_(self):
        '''
        returns the dict of params required to recreate the layer
        '''
        input_dim, output_dim = self.W.shape
        config = {"input_dim": input_dim,
                  "output_dim": output_dim,
                  "init_method": self.init_method,
                  'trainable': self.trainable}
        base_config = super(Dense, self)._get_config_()
        return dict(list(base_config.items())+list(config.items()))


# adding aliases
Linear = Dense


class BN_mean(Layer):
    '''
    Mean only Batch normalization  Layer

    During Train:
        BN(x) = X - mean(X_batch) + beta

    During Test:
        BN(x) = X - mean_learned + beta
    '''

    def __init__(self, dim, *, elr=0.9, trainable=True):
        '''
        Initializes the BN_mean layer parameter
            beta is initialized to zero
            mean_learned is initialized to zero

        Args:
            dim (int): size of input passed
            elr (float): exponential learning rate for updating mean_learned
            trainable (bool): False parameters of the layer are frozen
                              True parameters are updated during optimizer step
        '''
        assert isinstance(elr, float), f'should be float'
        assert (0 < elr < 1), f'should be between 0 and 1 but given {elr}'
        self.dim = dim
        self.beta = np.zeros((1, int(np.prod(dim))))
        self.cache_in = None
        self.mean_learned = np.zeros_like(self.beta)
        self.elr = elr
        self.trainable = trainable
        self.l_name = 'Mean only Batchnorm'
        self.params = [self.beta]

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the BN_mean Layer

        Args:
            X (numpy.ndarray): Input array
            train (bool): Set to True to enable caching for backward step

        Returns:
            Out (numpy.ndarray): Output after applying BN_mean()
        '''
        X_shape = X.shape
        X_flat = X.reshape(X_shape[0], -1)
        if train:
            current_mean = np.mean(X_flat, axis=0, keepdims=True)
            out_flat = X_flat - current_mean + self.beta

            # update for mean_learned (exponential moving average no bias
            # correction since we are going to be trainig it sufficiently)
            self.mean_learned = (self.elr*self.mean_learned
                                 + current_mean*(1-self.elr))

        else:  # during test use mean_learned
            out_flat = X_flat - self.mean_learned + self.beta
        return out_flat.reshape(X_shape)

    def backward(self, dY):
        '''
        Performs a backward pass through the BN_mean Layer

        Args:
            dY (numpy.ndarray): Output gradient

        Returns:
            dX (numpy.ndarray): Input gradient
            var_grad_list (list):
                trainable = True: [(beta,dbeta)]
                trainable = False: [ ]
        '''
        dY_shape = dY.shape
        dY_flat = dY.reshape(dY_shape[0], -1)
        N, D = dY_flat.shape
        dx1 = dY_flat
        dx2 = np.ones((N, D))/N * -1 * np.sum(dY_flat, axis=0)
        dX_flat = dx1 + dx2
        dX = dX_flat.reshape(dY_shape)
        if self.trainable:
            dbeta = np.sum(dY_flat, axis=0, keepdims=True)
            return dX, [(self.beta, dbeta)]
        # if not trainable
        return dX, []

    def _get_config_(self):
        '''
        returns the dict of params required to recreate the layer
        '''
        config = {"dim": self.dim, "elr": self.elr,
                  "trainable": self.trainable}
        base_config = super(BN_mean, self)._get_config_()
        return dict(list(base_config.items())+list(config.items()))


class BN(Layer):
    '''
    Batch normalization  Layer (Full)

    During Train:
        BN(x) = gamma(X - mean(X_batch))/std(X_batch) + beta

    During Test:
        BN(x) = gamma((X - mean_learned)/Learned_std) + beta
    '''

    def __init__(self, dim, *, elr=0.9, trainable=True):
        '''
        Initializes the BN layer parameter
            beta is initialized to zeros
            gamma is initialized to zeros
            mean_learned is initialized to zeros
            var_learned is initialized to zeros

        Args:
            dim (int) : size of input passed
            elr (float) : exponential lr for updating mean_learned
            trainable (bool) : False -> layer is frozen
                               True -> updated during optimizer step

        '''
        assert isinstance(elr, float), f'should be float'
        assert (0 < elr < 1), f'should be between 0 and 1 but given {elr}'
        self.dim = dim
        self.beta = np.zeros((1, int(np.prod(dim))))
        self.gamma = np.zeros((1, int(np.prod(dim))))
        self.cache_in = None
        self.mean_learned = np.zeros_like(self.beta)
        self.var_learned = np.zeros_like(self.gamma)
        self.elr = elr
        self.trainable = trainable
        self.l_name = 'Batchnorm'
        self.eps = 1e-10  # to avoid division_by_zero error if var = 0
        self.params = [self.gamma, self.beta]

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the BN Layer

        Args:
            X (numpy.ndarray): Input array
            train (bool): Set to True to enable caching for backward step

        Returns:
            Out (numpy.ndarray): Output after applying BN() transformation
        '''
        X_shape = X.shape
        X_flat = X.reshape(X_shape[0], -1)
        if train:
            assert X_shape[0] > 1, "not supported in training mode"
            current_mean = np.mean(X_flat, axis=0)
            current_var = np.var(X_flat, axis=0)
            X_norm_flat = (X_flat - current_mean)/np.sqrt(current_var+self.eps)
            out_flat = self.gamma*X_norm_flat + self.beta

            # update for mean_learned and std_learned
            # (exponential moving average no bias currection
            # since we are going to be trainig it sufficiently)
            self.mean_learned = (self.elr*self.mean_learned
                                 + current_mean*(1-self.elr))
            self.var_learned = (self.elr*self.var_learned
                                + current_var*(1-self.elr))

            self.cache_in = (X_flat, current_mean, current_var, X_norm_flat)

        else:  # during test use mean_learned adn std_learned
            X_norm_flat = ((X_flat - self.mean_learned)
                           / np.sqrt(self.var_learned+self.eps))
            out_flat = self.gamma*X_norm_flat + self.beta
        return out_flat.reshape(X_shape)

    def backward(self, dY):
        '''
        Performs a backward pass through the BN Layer

        Args:
            dY (numpy.ndarray): Output gradient

        Returns:
            dX (numpy.ndarray): Input gradient after backpropagation
            var_grad_list (list):
                trainable = True: [(gamma,dgamma), (beta,dbeta)]
                trainable = False: []
        '''
        if self.cache_in is None:
            raise RuntimeError(
                f'Gradient cache not defined. When training the train '
                f'argument must be set to true in the forward pass.'
            )
        X_flat, current_mean, current_var, X_norm_flat = self.cache_in
        dY_shape = dY.shape
        # fatten dY
        dY_flat = dY.reshape(dY_shape[0], -1)
        N = dY_shape[0]
        X_mu = X_flat - current_mean
        inv_var = 1/np.sqrt(current_var+self.eps)

        dX_norm = dY_flat * self.gamma

        d_var = (np.sum(dX_norm*X_mu, axis=0)
                 * (-((current_var+self.eps)**(-3/2))/2))

        d_mu = (np.sum(dX_norm*(-inv_var), axis=0)
                + (1/N)*d_var*np.sum(-2*X_mu, axis=0))

        dX_flat = (dX_norm*inv_var)+(d_mu+2*d_var*X_mu)/N
        dX = dX_flat.reshape(dY_shape)
        if self.trainable:
            dbeta = np.sum(dY_flat, axis=0, keepdims=True)
            dgamma = np.sum(dY_flat*X_norm_flat, axis=0, keepdims=True)
            return dX, [(self.gamma, dgamma), (self.beta, dbeta)]
        # if not trainable
        return dX, []

    def _get_config_(self):
        '''
        returns the dict of params required to recreate the layer
        '''
        config = {"dim": self.dim, "elr": self.elr,
                  "trainable": self.trainable}
        base_config = super(BN, self)._get_config_()
        return dict(list(base_config.items())+list(config.items()))


class Flatten(Layer):
    '''
    Flatten layer
    takes a tensor and converts it to a matrix
    This layer usually acts as an interface between conv layer and dense layer
    '''

    def __init__(self):
        '''
        Initialization :
            Does nothing since nothing to initialize
        '''
        self.cache_in = None
        self.l_name = 'Flatten'
        self.params = []

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the Flatten Layer

        Args:
            :X (numpy.ndarray): Input array
            :train (bool): No effect of this layer

        Returns:
            :Out (numpy.ndarray): Output after flattening
        '''
        self.shape = X.shape
        out = X.reshape(self.shape[0], -1)
        return out

    def backward(self, dY):
        '''
        Performs a backward pass through the BN_mean Layer

        Args:
            :dY (numpy.ndarray): Output gradient

        Returns:
            :dX (numpy.ndarray): Input gradient after reshaping dY
            :var_grad_list (list): [], since layer is not trainable
        '''
        dX = dY.reshape(self.shape)
        return dX, []
