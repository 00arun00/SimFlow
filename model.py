from optimizers import Optimizer
from iterators import Iterator
from layers import Layer
from losses import Loss
import numpy as np
class Model(object):
    '''
    Represents a neural network with any combination of layers
    '''
    def __init__(self):
        '''
        Returns a new empty neural network with no layers or loss
        Args:
            learning_rate (float): Learning rate to be used for minibatch SGD
        '''
        self.layers = []
        self.loss = None

    def add_layer(self, layer):
        '''
        Adds a layer to the network in a sequential manner.
        The input to this layer will be the output of the last added layer
        or the initial inputs to the networks if this is the first layer added.
        Args:
            layer (Layer): An instantiation of a class that extends Layer
        '''
        assert isinstance(layer,Layer)
        self.layers.append(layer)

    def set_loss_fn(self, loss):
        '''
        Sets the loss fuction that the network uses for training
        Args:
            loss (Loss): An instantiation of a class that extends Loss
        '''
        assert isinstance(loss,Loss)
        self.loss = loss

    def predict(self, inputs, train=False):
        '''
        Calculates the output of the network for the given inputs.
        Args:
            inputs (numpy.ndarray): Inputs to the network
        Returns:
            (numpy.ndarray): Outputs of the last layer of the network.
        '''
        output = inputs
        for layer in self.layers:
            output = layer.forward(output, train=train)
        return output

    def _forward_backward_(self, inputs, labels):
        '''
        Calculates the loss of the network for the given inputs and labels
        returns the list of tuples with variables and their curresponding gradients
        Args:
            inputs (numpy.ndarray): Inputs to the network
            labels (numpy.ndarray): Int representation of the labels (eg. the third class is represented by 2)
        Returns:
            (float): The loss before updating the network
            (list of tuples): variables and their gradients
        '''
        vars_and_grads = []

        # Forward pass
        output = self.predict(inputs, train=True)

        # Backward pass
        loss, grad = self.loss.get_loss(output, labels)
        for layer in reversed(self.layers):
            grad, layer_var_grad = layer.backward(grad)
            vars_and_grads += layer_var_grad

        return loss, vars_and_grads

    def set_optimizer(self,optimizer):
        assert isinstance(optimizer,Optimizer)
        self.optimizer = optimizer

    def set_iterator(self,iterator):
        assert isinstance(iterator,Iterator)
        self.iterator = iterator

    def fit(self,Data,Labels,epochs=1,*,verbose=True,**kwargs):
        allowed_kwargs = {'optimizer','iterator'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to fit: ' + str(k))
            else:
                if k == 'optimizer':
                    self.set_optimizer(kwargs[k])
                if k == 'iterator':
                    self.set_iterator(kwargs[k])
        # self.__dict__.update(kwargs)
        if not hasattr(self,'optimizer'):
            raise NameError('Optimizer not defined')
        if not hasattr(self,'iterator'):
            raise NameError('Iterator not defined')
        for epoch in range(epochs):
            total_loss = 0
            for curr_Data,curr_Labels in self.iterator.get_iterator(Data,Labels):
                loss,vars_and_grads = self._forward_backward_(curr_Data,curr_Labels)
                self.optimizer.update_step(vars_and_grads)
                total_loss+=loss
            average_loss = total_loss/Data.shape[0]
            if verbose:
                prnt_tmplt = ('Epoch: {:3}, average train loss: {:0.3f}')
                print(prnt_tmplt.format(epoch, average_loss))

    def score(self,Data,Labels):
        assert hasattr(self,"loss"),'loss function not defined please set a loss function to score with'
        assert Data.shape[0]==Labels.shape[0],'Number of elements in data should same as number of elements in Label'
        scores = self.predict(Data)
        loss, _ = self.loss.get_loss(Data, Labels)
        pred = np.argmax(scores,axis=1)
        correct = np.sum(pred==Labels)
        n_inp = Data.shape[0]
        avg_loss = loss/n_inp
        accuracy = correct/n_inp
        return avg_loss,accuracy
