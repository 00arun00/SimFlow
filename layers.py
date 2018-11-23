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

    def __repr__(self):
        if hasattr(self,l_name):
            return f'{self.l_name} layer'
        else:
            return f'Layer'

class Dense(Layer):
    '''
    Dense / Linear Layer
    
    Represent a linear transformation Y = X*W + b
        :X: is an numpy.ndarray with shape (batch_size, input_dim)
        :W: is a trainable matrix with dimensions (input_dim, output_dim)
        :b: is a bias with dimensions (1, output_dim)
        :Y: is an numpy.ndarray with shape (batch_size, output_dim)

    Initialization:
        :W: initialized with either Xavier or He initialization
        :b: initialized to zero

        Args:
            :input_dim (int): size of input passed
            :output_dim (int): size of output requred
            :init_method (str): initialization method to be used for Weights
            :trainable (bool):
                :False: parameters of the layer are frozed
                :True: parameters are updated during optimizer step
    '''
    def __init__(self, input_dim, output_dim,*,init_method='Xavier',trainable = True):
        '''
        Initializes the Desnse layer parameter
            :W: is initialized with either Xavier or He initialization
            :b: is initialized to zero

        Args:
            input_dim (int)   : size of input passed
            output_dim (int)  : size of output requred
            init_method (str) : initialization method to be used for Weights
            trainable (bool)  : if set to False parameters of the layer are frozed
                                if set to True parameters are updated during optimizer step
        '''
        self.init_method = init_method
        self.W = np.random.randn(input_dim, output_dim)
        self.W *= self._initializer_(self.W,init_method)
        self.b = np.zeros((1, output_dim))
        self.cache_in = None
        self.trainable = trainable
        self.l_name = 'Dense'

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the Dense Layer

        Args:
            :X (numpy.ndarray): Input array should be shape (batch_size x input_dim)
            :train (bool): Set to True to enable gardient caching for backward step

        Returns:
            :Out (numpy.ndarray): Output after applying transformation Y = X*W + b
                                  shape of output is (batch_size x output_dim)
        '''
        assert len(X.shape)==2,"input dimenstions not supported"
        assert X.shape[1]==self.W.shape[0],f"input dimension doesn't match, each X has dimension {X.shape[1]} but Weights defined are of shape {self.W.shape[0]}"
        out = X@self.W + self.b
        if train:
            self.cache_in = X
        return out

    def backward(self, dY):
        '''
        Performs a backward pass through the Dense Layer

        Args:
            :dY (numpy.ndarray): Output gradient backpropagated from layers in the front
                                  shape of dY is (batch_size x output_dim)

        Returns:
            :dX (numpy.ndarray): Input gradient after backpropagating dY through Dense layer
            :var_grad_list (list):
                :trainable = True: [(W,dW),(b,db)]
                :trainable = False: [ ]
        '''
        dX = dY@self.W.T
        if self.trainable:
            if self.cache_in is None:
                raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
            X = self.cache_in
            db = np.sum(dY, axis=0, keepdims=True)
            dW = X.T@dY
            assert X.shape == dX.shape,f"Dimensions of grad and variable should match, X has shape {X.shape} and dX has shape {dX.shape}"
            assert self.W.shape == dW.shape,f"Dimensions of grad and variable should match, W has shape {self.W.shape} and dW has shape {dW.shape}"
            assert self.b.shape == db.shape,f"Dimensions of grad and variable should match, b has shape {self.b.shape} and db has shape {db.shape}"
            return dX, [(self.W, dW), (self.b, db)]
        else:
            return dX, []

    def __repr__(self):
        return f'Dense Layer with shape {self.W.shape}'

#adding aliases
Linear = Dense

class ReLU(Layer):
    '''
    RelU layer
    Represent a nonlinear transformation Y = max(0,X)
    '''

    def __init__(self,*,trainable=True):
        '''
        Initialization :
            Does nothing since nothing to initialize
        '''
        self.cache_in = None
        self.trainable = trainable
        self.l_name = 'ReLU'

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the ReLU Layer

        Args:
            :X (numpy.ndarray): Input array
            :train (bool): Set to True to enable gardient caching for backward step

        Returns:
            :Out (numpy.ndarray): Output after applying transformation Y = max(0,X)
        '''
        out = np.maximum(X,0)
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
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        dX = dY*(self.cache_in>=0)
        return dX ,[]

# adding aliases
relu = ReLU

class Sigmoid(Layer):
    '''
    Sigmoid layer
    Represent a nonlinear transformation Y = 1/(1+e^(-X))
    '''

    def __init__(self,*,trainable=True):
        '''
        Initialization:
            Does nothing since nothing to initialize
        '''
        self.cache_in = None
        self.trainable = trainable
        self.l_name = 'Sigmoid'

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
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        out = self.cache_in
        dX = dY*(out*(1-out))
        return dX ,[]

class Tanh(Layer):
    '''
    Tanh layer
    Represent a nonlinear transformation Y = (1-e^(-2X))/(1+e^(-2X)) {tanh}
    '''
    def __init__(self,*,trainable=True):
        '''
        Initialization :
            Does nothing since nothing to initialize
        '''
        self.cache_in = None
        self.trainable = trainable
        self.l_name = 'Tanh'

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
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        out = self.cache_in
        dX = dY*(1-out**2)
        return dX ,[]

class BN_mean(Layer):
    '''
    Mean only Batch normalization  Layer

    During Train:
        BN(x) = X - mean(X_batch) + beta

    During Test:
        BN(x) = X - mean_learned + beta

    Initialization:
        :beta: initialized to zero
        :mean_learned: initialized to zero

    Args:
        :dim (int): size of input passed
        :elr (float): exponential learning rate for updating mean_learned
        :trainable (bool):
            :False: parameters of the layer are frozed
            :True: parameters are updated during optimizer step
    '''
    def __init__(self,dim,*,elr=0.9,trainable=True):
        '''
        Initializes the BN_mean layer parameter
            beta is initialized to zero
            mean_learned is initialized to zero

        Args:
            dim (int)         : size of input passed
            elr (float)       : exponential learning rate for updating mean_learned
            trainable (bool)  : if set to False parameters of the layer are frozed
                                if set to True parameters are updated during optimizer step
        '''
        assert isinstance(elr,float) and (elr>0 and elr<1), f'should be float value between 0 and 1 but given {elr}'
        self.beta = np.zeros((1,int(np.prod(dim))))
        self.cache_in = None
        self.mean_learned = np.zeros_like(self.beta)
        self.elr = elr
        self.trainable=trainable
        self.l_name = 'Mean only Batchnorm'

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the BN_mean Layer

        Args:
            :X (numpy.ndarray): Input array
            :train (bool): Set to True to enable caching for backward step

        Returns:
            :Out (numpy.ndarray): Output after applying BN_mean() transformation
        '''
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
        '''
        Performs a backward pass through the BN_mean Layer

        Args:
            :dY (numpy.ndarray): Output gradient backpropagated from layers in the front

        Returns:
            :dX (numpy.ndarray): Input gradient after backpropagating dY through BN_mean layer
            :var_grad_list (list):
                :trainable = True: [(beta,dbeta)]
                :trainable = False: [ ]
        '''
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
    Batch normalization  Layer (Full)

    During Train:
        BN(x) = gamma(X - mean(X_batch))/std(X_batch) + beta

    During Test:
        BN(x) = gamma((X - mean_learned)/Learned_std) + beta

    Initialization:
        :beta: initialized to zeros
        :gamma: initialized to zeros
        :mean_learned: initialized to zeros
        :var_learned: initialized to zeros

    Args:
        :dim (int): size of input passed
        :elr (float): exponential learning rate for updating mean_learned
        :trainable (bool):
            :False: parameters of the layer are frozed
            :True: parameters are updated during optimizer step
    '''
    def __init__(self,dim,*,elr=0.9,trainable=True):
        '''
        Initializes the BN layer parameter
            beta is initialized to zeros
            gamma is initialized to zeros
            mean_learned is initialized to zeros
            var_learned is initialized to zeros

        Args:
            dim (int)         : size of input passed
            elr (float)       : exponential learning rate for updating mean_learned
            trainable (bool)  : if set to False parameters of the layer are frozed
                                if set to True parameters are updated during optimizer step

        '''
        assert isinstance(elr,float) and (elr>0 and elr<1), f'should be float value between 0 and 1 but given {elr}'
        self.beta = np.zeros((1,int(np.prod(dim))))
        self.gamma = np.zeros((1,int(np.prod(dim))))
        self.cache_in = None
        self.mean_learned = np.zeros_like(self.beta)
        self.var_learned = np.zeros_like(self.gamma)
        self.elr = elr
        self.trainable=trainable
        self.l_name = 'Batchnorm'
        self.eps=1e-10 #to avoid division_by_zero error if var = 0

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the BN Layer

        Args:
            :X (numpy.ndarray): Input array
            :train (bool): Set to True to enable caching for backward step

        Returns:
            :Out (numpy.ndarray): Output after applying BN() transformation
        '''
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
        '''
        Performs a backward pass through the BN Layer

        Args:
            :dY (numpy.ndarray): Output gradient backpropagated from layers in the front

        Returns:
            :dX (numpy.ndarray): Input gradient after backpropagating dY through BN_mean layer
            :var_grad_list (list):
                :trainable = True: [(gamma,dgamma), (beta,dbeta)]
                :trainable = False: []
        '''
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
        out = X.reshape(self.shape[0],-1)
        return out

    def backward(self, dY):
        '''
        Performs a backward pass through the BN_mean Layer

        Args:
            :dY (numpy.ndarray): Output gradient backpropagated from layers in the front

        Returns:
            :dX (numpy.ndarray): Input gradient after reshaping dY
            :var_grad_list (list): [], since layer is not trainable
        '''
        dX = dY.reshape(self.shape)
        return dX,[]

class Conv2D(Layer):
    '''
    2D Convolutional Layer

    Initialization:
        :W: initialized with either Xavier or He initialization
        :b: initialized to zero

    Args:
        :outChannels (int):   Number of output channels
        :inChannels (int):   size of output requred
        :filter_size (int):   Size of each kernel (filter_size x filter_size)
        :stride (int):   Stride to be used
        :padding (int):   Padding to be used for convolution
        :init_method (str):   initialization method to be used for Weights
        :trainable (bool):
            :False: parameters of the layer are frozed
            :True: parameters are updated during optimizer step
    '''
    def __init__(self,outChannels,inChannels,filter_size,stride=1,padding=0,*,init_method='Xavier',trainable=False):
        '''
        Initializes the Convolutional layer
            W is initialized with either Xavier or He initialization
            b is initialized to zero

        Args:
            outChannels (int)   :   Number of output channels
            inChannels (int)    :   size of output requred
            filter_size (int)   :   Size of each kernel (filter_size x filter_size)
            stride (int)        :   Stride to be used
            padding (int)       :   Padding to be used for convolution
            init_method (str)   :   initialization method to be used for Weights
            trainable (bool)    :   if set to False parameters of the layer are frozed
                                    if set to True parameters are updated during optimizer step
        '''
        assert isinstance(outChannels,int) and outChannels > 0
        assert isinstance(inChannels,int) and inChannels > 0
        assert isinstance(filter_size,int) and filter_size > 0
        assert isinstance(stride,int) and stride > 0
        assert isinstance(padding,int) and padding >= 0

        self.cache_in = None
        W_size = (outChannels,inChannels,filter_size,filter_size) #currently supports only square kernels
        self.W = np.random.rand(*W_size)
        self.W *= self._initializer_(self.W,init_method)
        self.b = np.zeros((outChannels,1))
        self.stride = stride
        self.padding = padding
        self.trainable = trainable
        self.l_name = 'Conv2D'

    def __repr__(self):
        return f'Conv2D Layer with {self.W.shape[0]} number of filters of shape {self.W.Shape[1:]}, Stide = {self.stride}, padding = {self.padding} Trainable = {self.trainable}'

    @staticmethod
    def _convolve_(Input,kernel,bias=0,padding=0,stride=1):
        '''
        2D Convolution function:
        Convolves Inputs with the given kernel

        Args:
            Input (numpy.ndarray)     :    Input to be Convolved over
            kernel (numpy.ndarray)    :    Kernel to be Convoled with
            bias (numpy.ndarray)      :    bias optional, set to zero unless needed
            padding (int)             :    padding to be used
            stride (int)              :    stride to used

        Returns:
            out (numpy.ndarray)       :    Output after convolution
            Input_col (numpy.ndarray) :    retiled and stacked input
        '''
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
        '''
        Performs a forward pass through the 2D Convolution layer:
        Convolves Inputs with the Weights

        Args:
            :X (numpy.ndarray): Input array
            :train (bool): Set true to cache X and X_col

        Returns:
            :Out (numpy.ndarray): Output after Convolution
        '''
        output,X_col = self._convolve_(Input = X,kernel = self.W,bias = self.b,padding = self.padding, stride = self.stride)
        if train:
            self.cache_in = (X,X_col)
        return output

    def backward(self, dY):
        '''
        Performs a backward pass through the Conv2D Layer

        Args:
            :dY (numpy.ndarray): Output gradient backpropagated from layers in the front

        Returns:
            :dX (numpy.ndarray): Input gradient after backpropagating dY through Conv2D
            :var_grad_list (list):
                :trainable = True: [(W,dW), (b,db)]
                :trainable = False: [ ]
        '''
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

class dilated_Conv2D(Conv2D):
    '''
    2D Dilated Convolutional Layer

    Initialization:
        :W: initialized with either Xavier or He initialization
        :b: initialized to zero
        :dm: dilation matrix that is used to dilated the kernels

    Args:
        :outChannels (int):   Number of output channels
        :inChannels (int):   size of output requred
        :filter_size (int):   Size of each kernel (filter_size x filter_size)
        :dilation (int):   Dilation factor to be used
        :stride (int):   Stride to be used
        :padding (int):   Padding to be used for convolution
        :init_method (str):   initialization method to be used for Weights
        :trainable (bool):
            :False: parameters of the layer are frozed
            :True: parameters are updated during optimizer step
    '''
    def __init__(self,outChannels,inChannels,filter_size,dilation=2,stride=1,padding=0,*,init_method='Xavier',trainable=False):
        '''
        Initializes the Convolutional layer
            W is initialized with either Xavier or He initialization
            b is initialized to zero
            dm generates the dilation matrix that is used to dilated the kernels

        Args:
            outChannels (int)   :   Number of output channels
            inChannels (int)    :   size of output requred
            filter_size (int)   :   Size of each kernel (filter_size x filter_size)
            dilation (int)      :   Dilation factor to be used
            stride (int)        :   Stride to be used
            padding (int)       :   Padding to be used for convolution
            init_method (str)   :   initialization method to be used for Weights
            trainable (bool)    :   if set to False parameters of the layer are frozed
                                    if set to True parameters are updated during optimizer step
        '''
        super(dilated_Conv2D,self).__init__(outChannels=outChannels,inChannels=inChannels,filter_size=filter_size,stride=stride,padding=padding,trainable=trainable,init_method=init_method)
        assert isinstance(dilation,int) and dilation>0
        self.dilation = dilation #currently supports only symmetical dilations
        self.dm = self._create_dilation_mat_()
        self.l_name = 'dilated_Conv2D'
    def __repr__(self):
        return f'Conv2D Layer with {self.W.shape[0]} dilation = {dilation} number of filters of shape {self.W.Shape[1:]}, Stide = {self.stride}, padding = {self.padding} Trainable = {self.trainable}'

    def _create_dilation_mat_(self):
        '''
        private:
        generates a dilation matrix that is used to dilate the kernel

        Returns:
            dilation_mat (numpy.ndarray) : Matrix that is used to dilate the kernel
        '''
        I = np.eye(self.W.shape[2])
        z = np.zeros((1,self.W.shape[2]))
        res =[]
        for i in range(self.W.shape[2]):
            res.append(I[i])
            for k in range(self.dilation-1):
                res.append(z)
        res = np.row_stack(res)
        dilation_mat = res[:-self.dilation+1]
        return dilation_mat

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the dialted Convolution 2D layer:
        Convolves Inputs with the kernels after dilation

        Args:
            :X (numpy.ndarray):   Input array
            :train (bool):   Set true to cache X and X_col

        Returns:
            :Output (numpy.ndarray):   Output after Convolution
        '''
        # dilate the kernels using dilation matrix
        self.W_exp = self.dm@self.W@self.dm.T
        output,_ = self._convolve_(Input=X,kernel=self.W_exp,bias=self.b,padding= self.padding,stride = self.stride)
        if train:
            self.cache_in = X
        return output

    def backward(self, dY):
        '''
        Performs a backward pass through the dilated Conv2D Layer

        Args:
            :dY (numpy.ndarray): Output gradient backpropagated from layers in the front

        Returns:
            :dX (numpy.ndarray): Input gradient after backpropagating dY through dilated Conv2D
            :var_grad_list (list):
                :trainable = True: [(W,dW), (b,db)]
                :trainable = False: [ ]
        '''
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
