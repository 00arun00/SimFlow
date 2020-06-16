import numpy as np


class Iterator:
    '''
    Class representing Iterator

    Args:
        :batch_size (int): batch size to used
        :shuffle (bool): shuffles each time if set to True
    '''

    def __init__(self, batch_size=128, *, shuffle=True, method='direct'):
        '''
        Initializes the Iterator
        Args:
            batch_size (int) : batch size to used
            shuffle (bool)   : shuffles each time if set to True
        '''
        assert isinstance(batch_size, int), f'batch_size should be an integer'
        assert batch_size > 0, f'batch_size should be  > 0'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.method = method

    def get_iterator(self, Data, Labels):
        """
        Creates a generator to iterate through the Data

        Args:
            :Data (numpy.ndarray): Input to be iterated over
            :Labels (numpy.ndarray): Labesl to be iterated over

        Returns:
            :generator (generator): return a generator that
                can be used to iterate over Data and Labels
        """
        n_train = Data.shape[0]
        assert n_train == Labels.shape[0], 'len(Data) not same as len(Labels)'
        if self.method == 'full_batch':
            self.batch_size = n_train
        if self.shuffle:
            order = np.random.permutation(n_train)
        else:
            order = np.arange(n_train)
        start_idx = 0
        while start_idx < n_train:
            end_idx = min(start_idx+self.batch_size, n_train)
            idxs = order[start_idx:end_idx]
            mb_inputs = Data[idxs]
            mb_labels = Labels[idxs]
            yield mb_inputs, mb_labels
            start_idx += self.batch_size

    def __repr__(self):
        return f'{self.method} iterator with shuffling = {self.shuffle}'


class minibatch_iterator(Iterator):
    '''
    mini_batch iterator

    Args:
        :batch_size (int): batch size to used
        :shuffle (bool): shuffles each time if set to True
    '''

    def __init__(self, batch_size=128, *, shuffle=True):
        '''
        Initializes the Iterator
        Args:
            batch_size (int) : batch size to used
            shuffle (bool)   : shuffles each time if set to True
        '''
        super(minibatch_iterator, self).__init__(batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 method='mini_batch')

    def __repr__(self):
        return (f'mini_batch iterator '
                f'with batch_size = {self.batch_size} '
                f'and shuffling = {self.shuffle} ')


class fullbatch_iterator(Iterator):
    '''
    Full batch iterator

    Args:
        :shuffle (bool): shuffles each time if set to True
    '''

    def __init__(self, *, shuffle=True):
        '''
        Initializes the Iterator
        Args:
            shuffle (bool)   : shuffles each time if set to True
        '''
        super(fullbatch_iterator, self).__init__(batch_size=-1,
                                                 shuffle=shuffle,
                                                 method='full_batch')


class stochastic_iterator(Iterator):
    '''
    Stochastic iterator

    Args:
        :shuffle (bool): shuffles each time if set to True
    '''

    def __init__(self, *, shuffle=True):
        '''
        Initializes the Iterator
        Args:
            shuffle (bool): shuffles each time if set to True
        '''
        super(stochastic_iterator, self).__init__(batch_size=1,
                                                  shuffle=shuffle,
                                                  method='stochastic')
