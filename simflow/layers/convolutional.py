import numpy as np

from simflow.utils import im2col
from simflow.layers.layer_class import Layer


class Conv2D(Layer):
    '''
    2D Convolutional Layer
    '''

    def __init__(self, outChannels, inChannels, filter_size, stride=1,
                 padding=0, *, init_method='Xavier', trainable=False):
        '''
        Initialization:
            W: initialized with either Xavier or He initialization
            b: initialized to zero

        Args:
            outChannels (int): Number of output channels
            inChannels (int): size of output requred
            filter_size (int): Size of each kernel (n x n)
            stride (int): Stride to be used
            padding (int): Padding to be used for convolution
            init_method (str): initialization method to be used for Weights
            trainable (bool):
                False: parameters of the layer are frozed
                True: parameters are updated during optimizer step
        '''
        assert isinstance(outChannels, int) and outChannels > 0
        assert isinstance(inChannels, int) and inChannels > 0
        assert isinstance(filter_size, int) and filter_size > 0
        assert isinstance(stride, int) and stride > 0
        assert isinstance(padding, int) and padding >= 0

        self.cache_in = None
        # currently supports only square kernels
        self.W_size = (outChannels, inChannels, filter_size, filter_size)
        self.W = np.random.rand(*self.W_size)
        self.W *= self._initializer_(self.W, init_method)
        self.b = np.zeros((outChannels, 1))
        self.stride = stride
        self.padding = padding
        self.trainable = trainable
        self.l_name = 'Conv2D'
        self.params = [self.W, self.b]

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the 2D Convolution layer:
        Convolves Inputs with the Weights

        Args:
            X (numpy.ndarray): Input array
            train (bool): Set true to cache X and X_col

        Returns:
            Out (numpy.ndarray): Output after Convolution
        '''
        output, X_col = self._convolve_(
            Input=X, kernel=self.W, bias=self.b,
            padding=self.padding, stride=self.stride
        )
        if train:
            self.cache_in = (X, X_col)
        return output

    def backward(self, dY):
        '''
        Performs a backward pass through the Conv2D Layer

        Args:
            dY (numpy.ndarray): Output gradient backpropagated from layers

        Returns:
            dX (numpy.ndarray): Input gradient after backprop dY through Conv2D
            var_grad_list (list):
                trainable = True [(W,dW), (b,db)]
                trainable = False [ ]
        '''
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined')
        X, X_col = self.cache_in
        n_filter, d_filter, h_filter, w_filter = self.W.shape

        dY_reshaped = dY.transpose(1, 2, 3, 0).reshape(n_filter, -1)

        W_reshape = self.W.reshape(n_filter, -1)
        dX_col = W_reshape.T @ dY_reshaped
        dX = im2col.col2im_indices(dX_col, X.shape, h_filter, w_filter,
                                   padding=self.padding, stride=self.stride)
        assert X.shape == dX.shape, 'shape missmatch'
        if self.trainable:

            db = np.sum(dY, axis=(0, 2, 3)).reshape(n_filter, -1)

            dW = dY_reshaped @ X_col.T
            dW = dW.reshape(self.W.shape)

            assert self.W.shape == dW.shape, 'shape missmatch'
            assert self.b.shape == db.shape, 'shape missmatch'

            return dX, [(self.W, dW), (self.b, db)]
        # if not trainable
        return dX, []

    def __repr__(self):
        return (f'Conv2D Layer with {self.W.shape[0]}'
                f' number of filters of shape {self.W.Shape[1:]},'
                f' Stide = {self.stride}, padding = {self.padding}'
                f' Trainable = {self.trainable}')

    @staticmethod
    def _convolve_(Input, kernel, bias=0, padding=0, stride=1):
        '''
        2D Convolution function:
        Convolves Inputs with the given kernel

        Args:
            Input (numpy.ndarray): Input to be Convolved over
            kernel (numpy.ndarray): Kernel to be Convoled with
            bias (numpy.ndarray): bias optional, set to zero unless needed
            padding (int): padding to be used
            stride (int): stride to used

        Returns:
            out (numpy.ndarray): Output after convolution
            Input_col (numpy.ndarray): retiled and stacked input
        '''
        assert len(Input.shape) == 4 and len(kernel.shape) == 4
        n_x, d_x, h_x, w_x = Input.shape
        n_filter, d_filter, h_filter, w_filter = kernel.shape
        assert d_x == d_filter, "inputs not alligned for standard convolution"

        # check for validity of convolution
        h_out = (h_x - h_filter + 2 * padding) / stride + 1
        w_out = (w_x - w_filter + 2 * padding) / stride + 1

        if not h_out.is_integer() or not w_out.is_integer():
            raise Exception('Invalid output dimension!')

        h_out, w_out = int(h_out), int(w_out)

        Input_col = im2col.im2col_indices(Input, h_filter,
                                          w_filter, padding=padding,
                                          stride=stride)
        kernel_row = kernel.reshape(n_filter, -1)
        out = kernel_row @ Input_col + bias
        out = out.reshape(n_filter, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        return out, Input_col

    def _get_config_(self):
        '''
        returns the dict of params required to recreate the layer
        '''
        outChannels, inChannels, filter_size, _ = self.W.shape
        config = {"outChannels": outChannels, "inChannels": inChannels,
                  "stride": self.stride, "padding": self.padding,
                  "filter_size": filter_size, "trainable": self.trainable}
        base_config = super(Conv2D, self)._get_config_()
        return dict(list(base_config.items())+list(config.items()))


class dilated_Conv2D(Conv2D):
    '''
    2D Dilated Convolutional Layer
    '''

    def __init__(self, outChannels, inChannels, filter_size, dilation=2,
                 stride=1, padding=0, *, init_method='Xavier',
                 trainable=False):
        '''
        Initializes the Convolutional layer
            W is initialized with either Xavier or He initialization
            b is initialized to zero
            dm generates the dilation matrix that is used for dilations

        Args:
            outChannels (int): Number of output channels
            inChannels (int): size of output requred
            filter_size (int): Size of each kernel (filter_size x filter_size)
            dilation (int): Dilation factor to be used
            stride (int): Stride to be used
            padding (int): Padding to be used for convolution
            init_method (str): initialization method to be used for Weights
            trainable (bool): False -> the layer are frozen
                              True -> updated during optimizer step
        '''
        super(dilated_Conv2D, self).__init__(
            outChannels=outChannels, inChannels=inChannels,
            filter_size=filter_size, stride=stride, padding=padding,
            trainable=trainable, init_method=init_method
        )
        assert isinstance(dilation, int) and dilation > 0
        # currently supports only symmetical dilations
        self.dilation = dilation
        self.dm = self._create_dilation_mat_()
        self.l_name = 'dilated_Conv2D'

    def forward(self, X, train=True):
        '''
        Performs a forward pass through the dialted Convolution 2D layer:
        Convolves Inputs with the kernels after dilation

        Args:
            X (numpy.ndarray): Input array
            train (bool): Set true to cache X and X_col

        Returns:
            Output (numpy.ndarray): Output after Convolution
        '''
        # dilate the kernels using dilation matrix
        self.W_exp = self.dm@self.W@self.dm.T
        output, _ = self._convolve_(Input=X, kernel=self.W_exp, bias=self.b,
                                    padding=self.padding, stride=self.stride)
        if train:
            self.cache_in = X
        return output

    def backward(self, dY):
        '''
        Performs a backward pass through the dilated Conv2D Layer

        Args:
            dY (numpy.ndarray): Output gradient backpropagated

        Returns:
            dX (numpy.ndarray): Input grad after backprop
            var_grad_list (list):
                :trainable = True: [(W,dW), (b,db)]
                :trainable = False: [ ]
        '''
        if self.cache_in is None:
            raise RuntimeError(
                f'Gradient cache not defined. When training '
                f'the train argument must be set to true in the forward pass.'
            )
        X = self.cache_in

        n_filter, d_filter, h_filter, w_filter = self.W.shape

        exchange_mat = np.rot90(np.eye(h_filter))
        reversed_w = exchange_mat@self.W@exchange_mat.T
        dialated_reversed_w = self.dm@ reversed_w @self.dm.T

        dX, _ = self._convolve_(Input=dY, kernel=dialated_reversed_w.transpose(
            1, 0, 2, 3), padding=dialated_reversed_w.shape[2]-1)
        assert X.shape == dX.shape

        if self.trainable:
            db = np.sum(dY, axis=(0, 2, 3)).reshape(n_filter, -1)
            dW, _ = self._convolve_(Input=X.transpose(1, 0, 2, 3),
                                    kernel=dY.transpose(1, 0, 2, 3),
                                    stride=self.dilation, padding=0)
            dW = dW.transpose(1, 0, 2, 3)

            assert self.W.shape == dW.shape
            assert self.b.shape == db.shape
            return dX, [(self.W, dW), (self.b, db)]
        # if not trainable
        return dX, []

    def _create_dilation_mat_(self):
        '''
        private:
        generates a dilation matrix that is used to dilate the kernel

        Returns:
            dilation_mat (np.ndarray) : Matrix that is used for dilation
        '''
        I_ = np.eye(self.W.shape[2])
        z = np.zeros((1, self.W.shape[2]))
        res = []
        for i in range(self.W.shape[2]):
            res.append(I_[i])
            for k in range(self.dilation-1):
                res.append(z)
        res = np.row_stack(res)
        dilation_mat = res[:-self.dilation+1]
        return dilation_mat

    def _get_config_(self):
        '''
        returns the dict of params required to recreate the layer
        '''
        outChannels, inChannels, filter_size, _ = self.W.shape
        config = {"dilation": self.dilation}
        base_config = super(Conv2D, self)._get_config_()
        return dict(list(base_config.items())+list(config.items()))

    def __repr__(self):
        return (f'Conv2D Layer with {self.W.shape[0]} '
                f'dilation = {self.dilation} number of filters of '
                f'shape {self.W.Shape[1:]}, Stide = {self.stride}, '
                f'padding = {self.padding} Trainable = {self.trainable}')
