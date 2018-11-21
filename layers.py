import numpy as np
from im2col import *
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

class BN(Layer):
    '''
    Represents a Batch normalization  Layer (BN)
        During Train
        BN(x) = gamma(X - mean(X_batch))/std(X_batch) + beta

        During Test
        BN(x) = gamma((X - Learned_mean)/Learned_std) + beta

    '''
    def __init__(self,dim,*,exponential_learning_rate=0.9,trainable=True,eps=1e-10):
        self.beta = np.zeros((1,int(np.prod(dim))))
        self.gamma = np.zeros((1,int(np.prod(dim))))
        self.cache_in = None
        self.mean_learned = np.zeros_like(self.beta)
        self.var_learned = np.zeros_like(self.gamma)
        self.elr = exponential_learning_rate
        self.trainable=trainable
        self.eps=eps

    def forward(self, X, train=True):
        X_shape = X.shape
        X_flat = X.reshape(X_shape[0],-1)
        if train:
            assert X_shape[0]>1,"Batch_norm layer is not supported in training mode"
            current_mean = np.mean(X_flat,axis=0)
            current_var = np.var(X_flat,axis=0)
            X_norm_flat = (X_flat - current_mean)/np.sqrt(current_var+self.eps)
            out_flat = self.gamma*X_norm_flat + self.beta

            #update for mean_learned and std_learned (exponential moving average no bias currection since we are going to be trainig it sufficiently)
            self.mean_learned = (self.elr*self.mean_learned) + (current_mean*(1-self.elr))
            self.var_learned = (self.elr*self.var_learned) + (current_var*(1-self.elr))

            self.cache_in=(X_flat,current_mean,current_var,X_norm_flat)

        else: #during test use mean_learned adn std_learned
            X_norm_flat =(X_flat - self.mean_learned)/np.sqrt(self.var_learned+self.eps)
            out_flat =self.gamma*X_norm_flat  + self.beta
        return out_flat.reshape(X_shape)

    def backward(self, dY):
        if self.cache_in is None:
          raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        X_flat,current_mean,current_var,X_norm_flat = self.cache_in
        dY_shape = dY.shape
        #fatten dY
        dY_flat = dY.reshape(dY_shape[0],-1)
        N= dY_shape[0]
        X_mu = X_flat - current_mean
        inv_var = 1/np.sqrt(current_var+self.eps)

        dX_norm = dY_flat * self.gamma

        d_var = np.sum(dX_norm*X_mu,axis=0)*(-((current_var+self.eps)**(-3/2))/2)

        d_mu = np.sum(dX_norm*(-inv_var),axis=0) + (1/N)*d_var*np.sum(-2*X_mu,axis=0)

        dX_flat = (dX_norm*inv_var)+(d_mu+2*d_var*X_mu)/N
        dX = dX_flat.reshape(dY_shape)
        if self.trainable:
          dbeta = np.sum(dY_flat, axis=0, keepdims=True)
          dgamma = np.sum(dY_flat*X_norm_flat,axis=0,keepdims=True)
          return dX, [(self.gamma,dgamma),(self.beta, dbeta)]
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

class Conv(Layer):
    def __init__(self,outChannels,inChannels,filter_size,stride=1,padding=0,*,trainable=False):
        '''
        Represents a Convolutional layer
        '''
        self.cache_in = None
        W_size = (outChannels,inChannels,filter_size,filter_size) #currently supports only square kernels
        self.W = np.random.rand(*W_size)/filter_size #xavier initialization
        self.b = np.zeros((outChannels,1))
        self.stride = stride
        self.padding = padding
        self.trainable = trainable

    @staticmethod
    def _convolve_(Input,kernel,bias=0,padding=0,stride=1):
        assert len(Input.shape)==4 and len(kernel.shape)==4
        n_x, d_x, h_x, w_x = Input.shape
        n_filter, d_filter, h_filter, w_filter = kernel.shape
        assert d_x == d_filter,"inputs not alligned for standard convolution"

        #check for validity of convolution
        h_out = (h_x - h_filter + 2 * padding) / stride + 1
        w_out = (w_x - w_filter + 2 * padding) / stride + 1

        if not h_out.is_integer() or not w_out.is_integer():
            raise Exception('Invalid output dimension!')

        h_out, w_out = int(h_out), int(w_out)

        Input_col = im2col_indices(Input, h_filter, w_filter, padding=padding, stride=stride)
        kernel_row = kernel.reshape(n_filter, -1)
        out = kernel_row @ Input_col + bias
        out = out.reshape(n_filter, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        return out,Input_col


    def forward(self, X, train=True):
        output,X_col = self._convolve_(Input = X,kernel = self.W,bias = self.b,padding = self.padding, stride = self.stride)
        if train:
            self.cache_in = (X,X_col)
        return output

    def backward(self, dY):
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        X,X_col = self.cache_in
        n_filter, d_filter, h_filter, w_filter = self.W.shape

        dY_reshaped = dY.transpose(1, 2, 3, 0).reshape(n_filter, -1)

        W_reshape = self.W.reshape(n_filter, -1)
        dX_col = W_reshape.T @ dY_reshaped
        dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=self.padding, stride=self.stride)
        assert X.shape == dX.shape,'shape missmatch'
        if self.trainable:

            db = np.sum(dY, axis=(0, 2, 3)).reshape(n_filter, -1)

            dW = dY_reshaped @ X_col.T
            dW = dW.reshape(self.W.shape)


            assert self.W.shape == dW.shape,'shape missmatch'
            assert self.b.shape == db.shape,'shape missmatch'

            return dX, [(self.W,dW),(self.b,db)]
        else:
            return dX,[]

class dilated_Conv(Conv):

    def __init__(self,outChannels,inChannels,filter_size,dilation=2,stride=1,padding=0,*,trainable=False):
        '''
        Represents a Dialated Convolutional layer
        '''
        super(dilated_Conv,self).__init__(outChannels=outChannels,inChannels=inChannels,filter_size=filter_size,stride=stride,padding=padding,trainable=trainable)
        self.dilation = dilation #currently supports only symmetical dilations
        self.dm = self._create_dilation_mat_()

    def _create_dilation_mat_(self):
        I = np.eye(self.W.shape[2])
        z = np.zeros((1,self.W.shape[2]))
        res =[]
        for i in range(self.W.shape[2]):
            res.append(I[i])
            for k in range(self.dilation-1):
                res.append(z)
        res = np.row_stack(res)
        return res[:-self.dilation+1]

    def forward(self, X, train=True):
        self.W_exp = self.dm@self.W@self.dm.T
        output,_ = self._convolve_(Input=X,kernel=self.W_exp,bias=self.b,padding= self.padding,stride = self.stride)
        if train:
            self.cache_in = X
        return output

    def backward(self, dY):
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        X = self.cache_in

        n_filter, d_filter, h_filter, w_filter = self.W.shape

        exchange_mat= np.rot90(np.eye(h_filter))
        reversed_w = exchange_mat@self.W@exchange_mat.T
        dialated_reversed_w = self.dm@ reversed_w @self.dm.T

        dX,_ = self._convolve_(Input=dY,kernel=dialated_reversed_w.transpose(1,0,2,3),padding=dialated_reversed_w.shape[2]-1)
        assert X.shape == dX.shape

        if self.trainable:

            db = np.sum(dY, axis=(0, 2, 3)).reshape(n_filter, -1)
            dW,_ = self._convolve_(Input=X.transpose(1,0,2,3),kernel=dY.transpose(1,0,2,3),stride=self.dilation,padding=0)
            dW = dW.transpose(1,0,2,3)


            assert self.W.shape == dW.shape
            assert self.b.shape == db.shape
            return dX, [(self.W,dW),(self.b,db)]
        else:
            return dX,[]
