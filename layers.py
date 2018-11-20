import numpy as np

class Layer(object):
    '''
    Abstract class representing a neural network layer
    '''
    def forward(self, X, train=True):
        '''
        Calculates a forward pass through the layer.

        Args:
            X (numpy.ndarray): Input to the layer with dimensions (batch_size, input_size)

        Returns:
            (numpy.ndarray): Output of the layer with dimensions (batch_size, output_size)
        '''
        raise NotImplementedError('This is an abstract class')

    def backward(self, dY):
        '''
        Calculates a backward pass through the layer.

        Args:
            dY (numpy.ndarray): The gradient of the output with dimensions (batch_size, output_size)

        Returns:
            dX, var_grad_list
            dX (numpy.ndarray): Gradient of the input (batch_size, output_size)
            var_grad_list (list): List of tuples in the form (variable_pointer, variable_grad)
                where variable_pointer and variable_grad are the pointer to an internal
                variable of the layer and the corresponding gradient of the variable
        '''
        raise NotImplementedError('This is an abstract class')

    def _initializer_(self,W):
        """
        Initializer
        supports
        Xavier and he as of the moment
        """
        if self.init_method == 'Xavier':
            if len(W.shape)==2: #linear layer
              input_dim,output_dim = W.shape
              return np.sqrt(2.0/(input_dim+output_dim))
            elif len(W.shape)==4:#convolutional layer
              n_filter,d_filter,h_filter,w_filter = W.shape
              return np.sqrt(2.0/(h_filter*w_filter*d_filter))
            else:
              raise NotImplementedError('This W size is not defined')
        elif self.init_method == 'He':
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

class Dense(Layer):
    def __init__(self, input_dim, output_dim,*,init_method='Xavier',trainable = True):
        '''
        Represent a linear transformation Y = X*W + b
            X is an numpy.ndarray with shape (batch_size, input_dim)
            W is a trainable matrix with dimensions (input_dim, output_dim)
            b is a bias with dimensions (1, output_dim)
            Y is an numpy.ndarray with shape (batch_size, output_dim)
        W is initialized with Xavier-He initialization
        b is initialized to zero
        '''
        self.init_method = init_method
        self.W = np.random.randn(input_dim, output_dim)
        self.W *= self._initializer_(self.W)
        self.b = np.zeros((1, output_dim))
        self.cache_in = None
        self.trainable = trainable

    def forward(self, X, train=True):
        assert len(X.shape)==2,"input dimenstions not supported"
        assert X.shape[1]==self.W.shape[0],"input dimension doesn't match"
        out = X@self.W + self.b
        if train:
            self.cache_in = X
        return out

    def backward(self, dY):
        dX = dY@self.W.T
        if self.trainable:
          if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
          X = self.cache_in
          db = np.sum(dY, axis=0, keepdims=True)
          dW = X.T@dY
          assert X.shape == dX.shape,"Dimensions of grad and variable should match"
          assert self.W.shape == dW.shape,"Dimensions of grad and variable should match"
          assert self.b.shape == db.shape,"Dimensions of grad and variable should match"
          return dX, [(self.W, dW), (self.b, db)]
        else:
          return dX, []

class ReLU(Layer):

    def __init__(self,*,trainable=True):
      self.cache_in = None
      self.trainable = trainable

    def forward(self, X, train=True):
        out = np.maximum(X,0)
        if train:
            self.cache_in = X
        return out

    def backward(self, dY):
        if self.cache_in is None:
          raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        return dY*(self.cache_in>=0) ,[]


class sigmoid(Layer):

    def __init__(self,*,trainable=True):
      self.cache_in = None
      self.trainable = trainable

    def forward(self, X, train=True):
        out = 1/(1+np.exp(-X))
        if train:
            self.cache_in = out
        return out

    def backward(self, dY):
        if self.cache_in is None:
          raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        out = self.cache_in
        return dY*(out*(1-out)) ,[]

class tanh(Layer):

    def __init__(self,*,trainable=True):
      self.cache_in = None
      self.trainable = trainable

    def forward(self, X, train=True):
        out = np.tanh(X)
        if train:
            self.cache_in = out
        return out

    def backward(self, dY):
        if self.cache_in is None:
          raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        out = self.cache_in
        return dY*(1-out**2) ,[]

class BN_mean(Layer):
    '''
    Represents a mean only Batch normalization  Layer (BN)
        During Train
        BN(x) = X - mean(X_batch) + beta

        During Test
        BN(x) = X - Learned_mean + beta
    '''
    def __init__(self,dim,*,exponential_learning_rate=0.9,trainable=True):
        self.beta = np.zeros((1,int(np.prod(dim))))
        self.cache_in = None
        self.mean_learned = np.zeros_like(self.beta)
        self.elr = exponential_learning_rate
        self.trainable=trainable
    def forward(self, X, train=True):
        X_shape = X.shape
        X_flat = X.reshape(X_shape[0],-1)
        if train:
            current_mean = np.mean(X_flat,axis=0,keepdims=True)
            out_flat = X_flat - current_mean + self.beta

            #update for mean_learned (exponential moving average no bias currection since we are going to be trainig it sufficiently)
            self.mean_learned = (self.elr*self.mean_learned) + (current_mean*(1-self.elr))

        else: #during test use mean_learned
            out_flat = X_flat - self.mean_learned + self.beta
        return out_flat.reshape(X_shape)

    def backward(self, dY):
        dY_shape = dY.shape
        dY_flat = dY.reshape(dY_shape[0],-1)
        N,D = dY_flat.shape
        dx1 = dY_flat
        dx2 = np.ones((N,D))/N * -1 * np.sum(dY_flat, axis=0)
        dX_flat = dx1 + dx2
        dX = dX_flat.reshape(dY_shape)
        if self.trainable:
            dbeta = np.sum(dY_flat, axis=0, keepdims=True)
            return dX, [(self.beta, dbeta)]
        else:
            return dX,[]


class Flatten(Layer):
    '''
    Represents a flatten layer
    takes a tensor and converts it to a matrix
    '''
    def __init__(self):
        self.cache_in = None

    def forward(self, X, train=True):
        self.shape = X.shape
        out = X.reshape(self.shape[0],-1)
        return out

    def backward(self, dY):
        dX = dY.reshape(self.shape)
        return dX,[]
