import numpy as np

from simflow.optimizers import Optimizer
from simflow.iterators import Iterator
from simflow.layers import Layer
from simflow.losses import Loss


class Model(object):
    '''
    Represents a neural network with any combination of layers
    '''

    def __init__(self):
        '''
        Returns a new empty neural network with no layers or loss
        '''
        self.layers = []
        self.loss = None

    def summary(self):
        '''
        Prints a table describing the model layers and parameters used
        '''
        # currently not the most elegant implementation
        dash = "-"*75
        print(dash)
        print('{:<30s}|{:22s}|{:20s}'.format("Name", "Trainable Parameters",
                                             "Total Parameters"))
        print(dash)
        total_params = 0
        total_trainable_params = 0

        for layer in self.layers:
            param_size = 0
            trainable_param_size = 0
            for p in layer.params:
                param_size += np.prod(p.shape)
                if layer.trainable:
                    trainable_param_size += np.prod(p.shape)
            print('{:<30s}|{:>22d}|{:>20d}'.format(layer.l_name,
                                                   trainable_param_size,
                                                   param_size))
            total_params += param_size
            total_trainable_params += trainable_param_size
        print(dash)
        print('{:<30s}|{:>22d}|{:>20d}'.format("Total",
                                               total_trainable_params,
                                               total_params))
        print(dash)

    def get_params(self):
        '''
        Retruns the list of params in the model
        '''
        param_list = []
        for layer in self.layers:
            param_list += [(layer.l_name, layer.get_params())]
        return param_list

    def set_params(self, params):
        '''
        Sets the params of a layer with a new params

        Ags:
            :params (list of numpy.ndarray): new weights
        '''
        old_params = self.get_params()
        assert len(old_params) == len(params), 'Length missmatch'
        assert all((new_param[0] == old_param[0] and all(n.shape == o.shape)
                    for n, o in zip(new_param, old_param))
                   for (new_param, old_param)
                   in zip(params, old_params)), 'Structure missmatch'
        for (new_param, layer) in zip(params, self.layers):
            layer.set_params(new_param[1])

    def add_layer(self, layer):
        '''
        Adds a layer to the network in a sequential manner.
        The input to this layer will be the output of the last added layer
        or the initial inputs to the networks if this is the first layer added.

        Args:
            :layer (Layer): Layer to be added to the model
        '''
        assert isinstance(layer, Layer)
        self.layers.append(layer)

    def set_loss_fn(self, loss):
        '''
        Sets the loss fuction that the network uses for training

        Args:
            :loss (Loss): Loss function to be used
        '''
        assert isinstance(loss, Loss)
        self.loss = loss

    def predict(self, inputs, train=False):
        '''
        Calculates the output of the network for the given inputs.

        Args:
            :inputs (numpy.ndarray): Inputs to the network
        Returns:
            :output (numpy.ndarray): Outputs of the last layer of the network.
        '''
        output = inputs
        for layer in self.layers:
            output = layer.forward(output, train=train)
        return output

    def _forward_backward_(self, inputs, labels):
        '''
        Calculates the loss of the network for the given inputs and labels

        Args:
            :inputs (numpy.ndarray): Inputs to the network
            :labels (numpy.ndarray): Int representation of the labels
                (eg. the third class is represented by 2)
        Returns:
            :loss (float): The loss before updating the network
            :vars_and_grads (list of tuples): variables and their gradients
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

    def set_optimizer(self, optimizer):
        '''
        Sets the optmizier to be used

        Args:
            :optimizer (Optimizer): Optimizer to be used
        '''
        assert isinstance(optimizer, Optimizer)
        self.optimizer = optimizer

    def set_iterator(self, iterator):
        '''
        Sets up the iterator

        Args:
            :iterator (Iterator): Iterator to be used
        '''
        assert isinstance(iterator, Iterator)
        self.iterator = iterator

    def fit(self, Data, Labels, epochs=1, *, verbose=True, **kwargs):
        '''
        Trains the model on the provided Data and Labels

        Updates the model parameters which are trainable

        Args:
            :Data (numpy.ndarray): Data to fit on
            :Labels (numpy.ndarray): Labels for the Data
            :epochs (int): Number of epochs to train
            :verbose (bool): Set to True to print progress, default True
        Kwargs:
            :optimizer(optimizer): Optimizer to be used
            :iterator (Iterator): Iterator to be used
        '''
        allowed_kwargs = {'optimizer', 'iterator'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to fit: ' + str(k))

            if k == 'optimizer':
                self.set_optimizer(kwargs[k])
            if k == 'iterator':
                self.set_iterator(kwargs[k])
        # self.__dict__.update(kwargs)
        if not hasattr(self, 'optimizer'):
            raise NameError('Optimizer not defined')
        if not hasattr(self, 'iterator'):
            raise NameError('Iterator not defined')
        for epoch in range(epochs):
            total_loss = 0
            for curr_Data, curr_Labels in self.iterator.get_iterator(Data,
                                                                     Labels):
                loss, vars_and_grads = self._forward_backward_(curr_Data,
                                                               curr_Labels)
                self.optimizer.update_step(vars_and_grads)
                total_loss += loss
            average_loss = total_loss/Data.shape[0]
            if verbose:
                prnt_tmplt = ('Epoch: {:3}, average train loss: {:0.3f}')
                print(prnt_tmplt.format(epoch, average_loss))

    def score(self, Data, Labels):
        '''
        Return loss and accuracy of a model on Data and Labels passed

        Args:
            :Data (numpy.ndarray): Data to find performance on
            :Labels (numpy.ndarray): Labels to find performance on
        '''
        assert hasattr(
            self, "loss"), 'please set a loss function to score with'
        assert Data.shape[0] == Labels.shape[0], 'should same'
        scores = self.predict(Data)
        loss, _ = self.loss.get_loss(scores, Labels)
        pred = np.argmax(scores, axis=1)
        correct = np.sum(pred == Labels)
        n_inp = Data.shape[0]
        avg_loss = loss/n_inp
        accuracy = correct/n_inp
        return avg_loss, accuracy
